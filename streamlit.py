import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Custom styling for the app
st.markdown(
    """
    <style>
        .stApp { background-color: #1ED760; color: black; }
        .stMarkdown, .css-1v0mbdj, .stTextInput, .stFileUploader, .stSelectbox, .stTextArea {
            color: black !important;
        }
        .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Efficient model loading
@st.cache_resource
def get_model():
    return tf.keras.models.load_model('Best_Model.h5')

model = get_model()

# App title and instructions
st.title(" Multi-class Weather Classifier ")
st.write(
    "Upload a weather image and it will classify it as **Cloudy**, **Rain**, **Shine**, or **Sunrise**."
)

# File uploader for images
uploaded_file = st.file_uploader(" Upload a weather image", type=["jpg", "jpeg", "png"])

def predict_weather(image_file, model):
    target_size = (75, 75)
    image = ImageOps.fit(image_file, target_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_batch)
    return prediction

# Main logic
if uploaded_file:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Uploaded Image', use_container_width=True)

    prediction = predict_weather(input_image, model)
    categories = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    label = categories[int(np.argmax(prediction))]
    confidence = float(np.max(prediction)) * 100

    st.success(f"🌤 **Classification:** {label}")
    st.info(f" **Model Confidence:** {confidence:.2f}%")
else:
    st.info(" Upload an image to begin.")
