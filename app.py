import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model_path = 'dr_model.keras'
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?id=19sMfDev7XMMuBadAOV1LPU2BXbrhexiw'
        gdown.download(url, model_path, quiet=False)
    # Load model with custom objects if any
    with tf.keras.utils.custom_object_scope({'Addons>CohenKappa': tf.keras.metrics.MeanSquaredError()}):
        model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ---------------------- PREDICTION FUNCTION ----------------------
def predict_image(image):
    image = image.resize((224, 224))  # Resize to model input shape
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:  # remove alpha if present
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

# ---------------------- CLASS NAMES ----------------------
class_names = {
    0: "No Diabetic Retinopathy",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative Diabetic Retinopathy"
}

# ---------------------- STREAMLIT UI ----------------------
st.title("👁️ Diabetic Retinopathy Classification")
st.write("Upload a retina image to detect the presence and severity of diabetic retinopathy.")

uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            prediction = predict_image(image)
            st.success(f"Prediction: **{class_names[prediction]}**")
