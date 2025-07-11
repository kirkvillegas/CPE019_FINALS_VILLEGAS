import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: white;
    }
    .stMarkdown, .css-1v0mbdj, .stTextInput, .stFileUploader, .stSelectbox, .stTextArea {
        color: white !important;
    }
    .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

#  Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('My_model.h5')
    return model

model = load_model()

#  Title
st.title(" Multi-class Weather Classifier ")
st.write("Upload a weather image and it will classify it as **Cloudy**, **Rain**, **Shine**, or **Sunrise**.")

#  File upload
file = st.file_uploader(" Upload a weather image", type=["jpg", "jpeg", "png"])

#  Prediction function
def import_and_predict(image_data, model):
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

#  Run model prediction
if file is None:
    st.info(" Upload an image to begin.")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    prediction = import_and_predict(image, model)
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🌤 **Classification:** {predicted_class}")
    st.info(f" **Model Confidence:** {confidence:.2f}%")import streamlit as st
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

#  Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('My_model.h5')
    return model

model = load_model()

#  Title
st.title(" Multi-class Weather Classifier ")
st.write("Upload a weather image and it will classify it as **Cloudy**, **Rain**, **Shine**, or **Sunrise**.")

#  File upload
file = st.file_uploader(" Upload a weather image", type=["jpg", "jpeg", "png"])

#  Prediction function
def import_and_predict(image_data, model):
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

#  Run model prediction
if file is None:
    st.info(" Upload an image to begin.")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    prediction = import_and_predict(image, model)
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🌤 **Classification:** {predicted_class}")
    st.info(f" **Model Confidence:** {confidence:.2f}%")
