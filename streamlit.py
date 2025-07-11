import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.markdown("""
    <style>
    .stApp {
        background-color: #1ED760;
        color: black;
    }
    .stMarkdown, .css-1v0mbdj, .stTextInput, .stFileUploader, .stSelectbox, .stTextArea {
        color: black !important;
    }
    .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = 'Best_Model.h5'

# Allow user to upload ANY type of file
uploaded_files = st.file_uploader(
    "Upload your files (model .h5 or images .jpg/.jpeg/.png)", 
    type=None, 
    accept_multiple_files=True
)

model = None
image_file = None

# Detect model and image file from uploaded files
if uploaded_files:
    for file in uploaded_files:
        if file.name.lower().endswith('.h5'):
            # Save model file
            with open(MODEL_PATH, "wb") as f:
                f.write(file.read())
            model = tf.keras.models.load_model(MODEL_PATH)
        elif file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_file = file

if not model:
    st.warning(f"Please upload your Keras .h5 model file.")
    st.stop()

# Title
st.title("Multi-class Weather Classifier")
st.write("Upload a weather image and it will classify it as **Cloudy**, **Rain**, **Shine**, or **Sunrise**.")

def import_and_predict(image_data, model):
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

if image_file is None:
    st.info("Upload an image to begin.")
else:
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    prediction = import_and_predict(image, model)
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🌤 **Classification:** {predicted_class}")
    st.info(f"**Model Confidence:** {confidence:.2f}%")
