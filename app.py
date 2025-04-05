import streamlit as st
import tensorflow as tf
import urllib.request
import os

# Constants
MODEL_URL = "https://www.dropbox.com/scl/fi/57dtrpmrpbn49wmqqeifs/dr_model.keras?rlkey=l6vadl5cr7hluawv3xqhc3ugy&dl=1"
MODEL_PATH = "dr_model.keras"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading model from Dropbox..."):
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error("❌ Failed to load the model. Please verify the model file format or URL.")
        st.exception(e)
        return None

# Load model
model = load_model()

# UI
st.title("🩺 Diabetic Retinopathy Detection")

if model:
    st.success("✅ Model loaded successfully!")
    uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0) / 255.0

        prediction = model.predict(image_array)
        st.write("🔍 Prediction:", prediction)
else:
    st.stop()
