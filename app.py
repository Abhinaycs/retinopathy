import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request

# Constants
MODEL_URL = "https://www.dropbox.com/scl/fi/57dtrpmrpbn49wmqqeifs/dr_model.keras?rlkey=l6vadl5cr7hluawv3xqhc3ugy&dl=1"
MODEL_PATH = "dr_model.keras"
IMG_SIZE = (224, 224)

# Load model with caching and error handling
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("🔄 Downloading model from Dropbox..."):
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error("❌ Failed to load the model. Please verify the model file format or URL.")
        st.exception(e)
        return None

# Image preprocessing
def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction logic
def predict(image, model):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return prediction

# Streamlit UI
st.title("👁️ Retinopathy Detection")
st.write("Upload a fundus image to detect signs of diabetic retinopathy.")

# Upload
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

# Load the model
model = load_model()

# Predict
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Making prediction..."):
        prediction = predict(image, model)

    st.success("✅ Prediction complete!")
    st.write("### Raw Prediction Output:", prediction)

    # Optional: Interpret the output based on your model's classes
    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    predicted_class = classes[np.argmax(prediction)]
    st.write(f"### Predicted Class: **{predicted_class}**")
