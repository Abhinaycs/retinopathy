import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="Retinopathy Detector", layout="centered")

MODEL_PATH = "model.h5"
CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

# Download model if not present
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1Yo5cF9VX3E1dD53D6S6xAtnmmrLbpibB"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.0
    return img_array

st.title("👁️ Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = float(np.max(predictions)) * 100

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
