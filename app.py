import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

# Constants
REPO_ID = "Abhinay2711/retinopathy"
FILENAME = "dr_model.keras"  # Make sure this matches the uploaded file name

# Load model using cache
@st.cache_resource
def load_model():
    try:
        with st.spinner("🔄 Downloading and loading model from Hugging Face..."):
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                use_auth_token="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"
            )
            model = tf.keras.models.load_model(model_path)
            return model
    except Exception as e:
        st.error("❌ Failed to load the model from Hugging Face.")
        st.exception(e)
        return None

# Image preprocessing
def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((224, 224))  # Resize as per your model input
    image = np.array(image) / 255.0   # Normalize if needed
    image = np.expand_dims(image, axis=0)
    return image

# App UI
st.set_page_config(page_title="🩺 Diabetic Retinopathy Detection", layout="centered")

st.title("🩺 Diabetic Retinopathy Detection")
st.write("Upload a retinal image and let the model predict the condition.")

uploaded_file = st.file_uploader("📤 Upload a retinal image", type=["jpg", "jpeg", "png"])

model = load_model()

if uploaded_file and model:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(image)

        with st.spinner("🧠 Making prediction..."):
            prediction = model.predict(processed_image)

        # Example logic (adjust according to your model output)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
        st.success(f"✅ Prediction: **{class_names[predicted_class]}** with **{confidence:.2f}%** confidence.")
    except Exception as e:
        st.error("⚠️ Error processing the image or making prediction.")
        st.exception(e)
elif not model:
    st.error("❌ Model could not be loaded. Check Hugging Face URL or token.")
