import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import os

MODEL_URL = "https://huggingface.co/Abhinay2711/retinopathy/resolve/main/dr_model.keras"
MODEL_PATH = "dr_model.keras"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading model from Hugging Face..."):
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error("❌ Failed to load the model from Hugging Face.")
        st.exception(e)
        return None

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.set_page_config(page_title="🩺 Diabetic Retinopathy Detection", layout="centered")
st.title("🩺 Diabetic Retinopathy Detection")
st.markdown("Upload a **retina image** to predict the presence and severity of diabetic retinopathy.")

uploaded_file = st.file_uploader("📤 Upload Retina Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retina Image", use_column_width=True)

    model = load_model()
    if model:
        st.markdown("🔍 **Analyzing the image...**")
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"**Prediction:** {CLASS_NAMES[class_index]}")
        st.info(f"**Confidence:** {confidence:.2%}")
