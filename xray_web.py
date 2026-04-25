
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Pneumonia Identification System
         """)

# --- Decision Threshold Slider ---
# This allows fine-tuning the confidence required to diagnose Pneumonia.
# Higher threshold = fewer false positives for Normal patients.
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Pneumonia Confidence Threshold (%)",
    min_value=50,
    max_value=95,
    value=50,  # Default: standard 50% (majority wins)
    step=5,
    help="Raise this value to require higher confidence before diagnosing Pneumonia. "
         "Useful if the model is biased toward the majority class."
)
threshold_fraction = threshold / 100.0

file = st.file_uploader("Please upload a chest scan file", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)

    # Apply softmax to get proper probabilities (model outputs logits)
    score = tf.nn.softmax(predictions[0]).numpy()

    class_names = ['Normal', 'Pneumonia']

    # --- Apply threshold-based classification ---
    pneumonia_prob = score[1]  # Probability of Pneumonia
    normal_prob = score[0]     # Probability of Normal

    if pneumonia_prob >= threshold_fraction:
        predicted_class = 'Pneumonia'
        confidence = pneumonia_prob * 100
    else:
        predicted_class = 'Normal'
        confidence = normal_prob * 100

    # Display result with color
    if predicted_class == 'Pneumonia':
        st.error(
            f"⚠️ This image most likely belongs to **{predicted_class}** "
            f"with a **{confidence:.2f}%** confidence."
        )
    else:
        st.success(
            f"✅ This image most likely belongs to **{predicted_class}** "
            f"with a **{confidence:.2f}%** confidence."
        )

    # Show detailed probabilities
    st.sidebar.subheader("Detailed Probabilities")
    st.sidebar.write(f"Normal: {normal_prob * 100:.2f}%")
    st.sidebar.write(f"Pneumonia: {pneumonia_prob * 100:.2f}%")
    st.sidebar.write(f"Threshold: {threshold}%")
