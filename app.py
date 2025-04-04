import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Retinopathy Detection", layout="centered")

# Title
st.title("👁️ Diabetic Retinopathy Detection")

# Download model if not present
MODEL_PATH = "retinopathy_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1TSHBXjMu6a4tFVHnLnz6AtP3y8g3ZVmT"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Class names
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# File uploader
uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    image = image.resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # Prediction
    prediction = model.predict(image_array)
    predicted_class = classes[np.argmax(prediction)]

    st.markdown(f"### 🔍 Predicted Class: **{predicted_class}**")
