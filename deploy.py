import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

# ==== Config ====
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
input_shape = (128, 128, 3)

# ==== Carica modello ====
@st.cache_resource
def load_effnet_model():
    return load_model("efficientnet_final_new.h5")  # <-- metti il tuo nome file

model = load_effnet_model()

# ==== Interfaccia Streamlit ====
st.title("Skin Lesion Classifier")
st.markdown("Scatta una foto della lesione per classificarla usando un modello EfficientNet.")

# 📷 Acquisizione da fotocamera
img_file = st.camera_input("Scatta o carica una foto")

if img_file is not None:
    # Preprocessing
    image = Image.open(img_file).convert("RGB")
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized).astype("float32")
    img_pre = preprocess_input(img_array)  # EfficientNet preprocessing
    img_batch = np.expand_dims(img_pre, axis=0)

    # Predizione
    preds = model.predict(img_batch)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    # Output
    st.image(image, caption="Immagine caricata", use_container_width=True)
    st.markdown(f"### ✅ Predizione: **{class_names[pred_class]}**")
    st.markdown(f"Confidenza: `{confidence:.2f}`")

    # Mostra probabilità per tutte le classi
    st.bar_chart({cls: float(prob) for cls, prob in zip(class_names, preds[0])})
