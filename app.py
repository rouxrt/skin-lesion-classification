import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import torch
import segmentation_models_pytorch as smp
from torchvision.utils import save_image  # solo se vuoi salvare
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense


# ==== Config ====
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
input_shape = (128, 128, 3)

# ==== Carica modello ====


@st.cache_resource
def load_custom_effnet_model(weights_path="efficientnet_weights.h5", input_shape=(224, 224, 3), num_classes=7):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = True  # hai fatto fine-tuning

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.load_weights(weights_path)
    return model

custom_model = load_custom_effnet_model("efficientnet_weights.h5")

@st.cache_resource
def load_unet_model(path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = load_unet_model("final_unet.pth")  # <-- metti il path giusto

test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def transform_image_for_unet(image_np):
    augmented = test_transform(image=image_np)
    return augmented['image']


# ==== DullRazor ====
def dullrazor_strong(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
    result = cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA)
    return result, hair_mask

def resize_and_normalize_image_from_array(img_array, size=(224, 224)):
    img_resized = cv2.resize(img_array, size, interpolation=cv2.INTER_LANCZOS4)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_to_display = (img_normalized * 255).astype(np.uint8)
    img_pil_out = Image.fromarray(img_to_display)
    return img_normalized, img_pil_out

# ==== Sharpen + CLAHE ====
soft_kernel = np.array([[0, -0.5, 0],
                        [-0.5, 3, -0.5],
                        [0, -0.5, 0]])

def apply_sharpen(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def apply_clahe(img, clipLimit=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def isolate_segmented_area(img_rgb, mask_np, target_size=(224, 224)):
    # Resize immagine e maschera se necessario
    if img_rgb.shape[:2] != target_size:
        img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    if mask_np.shape[:2] != target_size:
        mask_np = cv2.resize(mask_np, target_size, interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask_np >= 0.5).astype(bool)
    img_segmented = img_rgb.copy()
    img_segmented[~mask_bin] = 0  # tutto nero dove mask è 0
    return img_segmented


# ==== Interfaccia Streamlit ====
st.title("Skin Lesion Classifier")
st.markdown("Scatta o carica una foto della lesione per classificarla usando un modello EfficientNet. I peli vengono automaticamente rimossi.")

# ==== Scelta metodo input ====
img_file = st.file_uploader("Carica una foto", type=["jpg", "jpeg", "png"])

# ==== Elaborazione ====
if img_file is not None:
    try:
        pil_img = Image.open(img_file).convert("RGB")
    except Exception as e:
        st.error(f"Errore nel caricamento dell'immagine: {e}")
        st.stop()

    # 🔁 Conversione a JPEG (in memoria)
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")

    max_size = (224, 224)  # o (256, 256) per test
    pil_img = pil_img.resize(max_size)

    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 🔍 Mostra immagine originale
    st.image(img_rgb, caption="🖼️ Immagine originale", use_container_width=True)

    # --- Step 2: Rimozione peli (DullRazor) ---
    img_dullrazor, hair_mask = dullrazor_strong(img_bgr)
    img_final_rgb = cv2.cvtColor(img_dullrazor, cv2.COLOR_BGR2RGB)


    # 🔍 Mostra immagine dopo DullRazor
    #st.image(img_final_rgb, caption="🧼 Immagine senza peli (DullRazor)", use_container_width=True)

    

    # --- Step 4: Ridimensiona e normalizza a 224x224 (senza salvare) ---
    img_normalized, img_pil_out = resize_and_normalize_image_from_array(img_final_rgb, size=(224, 224))
    # === Step 3: Sharpen + CLAHE ===
    img_enhanced = apply_sharpen(img_dullrazor, soft_kernel)
    img_enhanced = apply_clahe(img_enhanced, clipLimit=2.0)

# Converti per visualizzazione e predizione
    img_final_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)

# 🔍 Mostra immagine migliorata
    # === Segmentazione con UNet ===
    img_for_unet = img_final_rgb  # RGB image (np.array)
    input_tensor = transform_image_for_unet(img_for_unet).unsqueeze(0).to(device)

    with torch.no_grad():
        output = unet_model(input_tensor)
        output = torch.sigmoid(output)
        mask = (output > 0.5).float()

# Converti maschera in immagine da visualizzare
    mask_np = mask.squeeze().cpu().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))

    segmented_np = isolate_segmented_area(img_final_rgb, mask_np, target_size=(224, 224))
    segmented_pil = Image.fromarray(segmented_np)

# Mostra
    st.image(segmented_pil, caption="🔬 Lesione isolata (segmentata)", use_container_width=True)


# Usa l'immagine segmentata PIL → numpy
    img_np = np.array(segmented_pil.resize((224, 224)))  # già isolata

# Preprocessa
    img_np = np.array(segmented_pil.resize((224, 224)))  # già segmentata e ritagliata
    img_array = preprocess_input(img_np.astype(np.float32))
    img_input = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

# Predizione
    preds = custom_model.predict(img_input)
    pred_class_idx = np.argmax(preds, axis=1)[0]
    pred_conf = float(np.max(preds))
    pred_label = class_names[pred_class_idx]

# 🔍 Mostra risultato
    st.markdown(f"### ✅ Predizione: **{pred_label}**")
    st.markdown(f"📊 Confidenza: **{pred_conf:.2%}**")

# Bar chart con tutte le probabilità
    prob_dict = {cls: float(prob) for cls, prob in zip(class_names, preds[0])}
    st.bar_chart(prob_dict)
