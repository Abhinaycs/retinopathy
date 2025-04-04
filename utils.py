from tensorflow.keras import models
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow_addons.metrics import CohenKappa

def load_model(model_path): 
    model = models.load_model(model_path,custom_objects={"CohenKappa": CohenKappa} )
    return model

def process_image(im, desired_size=224):
    im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)
    im = np.expand_dims(np.array(im),0)
    return im
 
def get_footer():
    footer = """
     <style>
    footer {
	visibility: hidden;
	}
    footer:after {
	content:'Contact: belyazidyous@gmail.com'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
    }
    </style>
    """
    return footer

