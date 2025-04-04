import streamlit as st
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from PIL import Image
import gdown
import tempfile

st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

st.title("👁️ Diabetic Retinopathy Classifier")

# Load the Keras model from Google Drive using gdown
@st.cache_resource
def load_model_from_gdrive():
    url = "https://drive.google.com/uc?id=19sMfDev7XMMuBadAOV1LPU2BXbrhexiw"  # Replace with your actual model ID

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
        gdown.download(url, tmp_file.name, quiet=False)
        tmp_file_path = tmp_file.name

    # Register custom metric from TensorFlow Addons
    custom_objects = {"Addons>CohenKappa": tfa.metrics.CohenKappa}

    model = tf.keras.models.load_model(tmp_file_path, custom_objects=custom_objects)
    return model

model = load_model_from_gdrive()

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust based on your model's input shape
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:  # Handle PNG with alpha
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Map predicted class to readable label
label_map = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Upload section
uploaded_file = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retina Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            preprocessed = preprocess_image(image)
            prediction = model.predict(preprocessed)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            st.success(f"🩺 **Prediction:** {label_map[predicted_class]}")
            st.info(f"🔍 **Confidence:** {confidence:.2%}")
