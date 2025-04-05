import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import urllib.request
import os

st.set_page_config(page_title="Retinopathy Detector", layout="centered")

# Constants
MODEL_PATH = "dr_model.keras"
MODEL_URL = "https://www.dropbox.com/scl/fi/57dtrpmrpbn49wmqqeifs/dr_model.keras?rlkey=l6vadl5cr7hluawv3xqhc3ugy&st=i5co4icd&dl=1"
CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

# Load and cache model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0
    return img_array

# Streamlit UI
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
