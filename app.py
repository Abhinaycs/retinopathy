import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
from tensorflow_addons.metrics import CohenKappa
from PIL import Image

@st.cache_resource
def load_model_from_gdrive():
    url = "https://drive.google.com/uc?id=1mMp6v2OR6uL2xA0C3YrEavUReLNEUpNo"
    output_path = "retinopathy_model.keras"
    gdown.download(url, output_path, quiet=False)

    model = tf.keras.models.load_model(
        output_path,
        custom_objects={"Addons>CohenKappa": CohenKappa}
    )
    return model

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # handle images with alpha channel
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def main():
    st.title("Diabetic Retinopathy Classification")
    st.write("Upload a retinal image to classify the stage of diabetic retinopathy.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner('Loading model and predicting...'):
            model = load_model_from_gdrive()
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
            predicted_class = class_names[np.argmax(prediction)]

        st.success(f"Prediction: **{predicted_class}**")

if __name__ == "__main__":
    main()
