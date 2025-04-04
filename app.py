# import numpy as np
# import streamlit as st
# from PIL import Image
# from utils import process_image, get_footer, load_model
# from copy import deepcopy

 
 

# def highlight_prediction(options, idx):
#     options = deepcopy(options)
#     highlight = f'''<span style="color:green">**{options[idx]}** </span>'''
#     options[idx] = highlight
#     return '<br>'.join(options)


# if __name__ == '__main__':
#     st.image("assets/istockphoto-1189207226-612x612.jpg")
#     st.markdown("<h1 style='text-align: center; color: black;'>"
#                 "<center>&emsp;&emsp;Detection of Diabetic Retinopathy </center></h1>", unsafe_allow_html=True)
#     st.markdown("<br><br><br><br>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("Upload a retina medical image...", type=["jpg","png"])
#     st.write(" ")
#     st.markdown(get_footer(), unsafe_allow_html=True)

#     if uploaded_file is not None:

#         options = ['No Diabetic Retinopathy', 'Mild', 'Moderate' 'Severe',
#                    'Profelivative Diabetic Retinopathy']
#         img_in = Image.open(uploaded_file)
#         img_in_processed = process_image(img_in)

#         col1, col2 = st.columns(2)
#         col1.image(img_in_processed)
#         st.write("")

#         model = load_model('assets/model_2021-08-30')
#         prediction = model.predict(img_in_processed).ravel()
#         idx = np.argmax(prediction)
#         col2.markdown("### Severity of Diabetic Retinopathy")
#         col2.markdown(highlight_prediction(options, idx), unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa  # Import TensorFlow Addons
from PIL import Image
import io

# Load the trained model with custom objects
@st.cache_resource
def load_model():
    model_path = "assets/densenet121_2025-04-04"  # Adjust if needed
    custom_objects = {"CohenKappa": tfa.metrics.CohenKappa}  # Register custom metric
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

model = load_model()

# Define class labels
CLASS_NAMES = ["No Diabetic Retinopathy", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Streamlit UI
st.title("Diabetic Retinopathy Classification")
st.write("Upload a fundus image to classify the severity of Diabetic Retinopathy.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    def preprocess_image(img):
        img = img.resize((224, 224))  # Resize to match model input size
        img = np.array(img) / 255.0   # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Display result
    st.subheader("Prediction:")
    st.write(f"Class: {CLASS_NAMES[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
    
    # Confidence-based recommendation
    if confidence > 0.7:
        st.success("High confidence in prediction.")
    else:
        st.warning("Low confidence. Consider consulting an ophthalmologist.")
