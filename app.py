import streamlit as st
import numpy as np
import tensorflow as tf
import os
import gdown
from PIL import Image

# ✅ Move this to the first line!
st.set_page_config(page_title="Potato Disease Detection", layout="centered")

# Function to set background image
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url}) no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Replace with your background image URL
background_url = "https://img.freepik.com/premium-photo/leaf-plant-pattern-illustration_728905-2516.jpg"  # Change this to your image link

# Apply background
set_background(background_url)

file_id = '1Cn_8SK7rq3RjQP1Ag37FSAZ_SCNryJB5'
url = "https://drive.google.com/file/d/1Cn_8SK7rq3RjQP1Ag37FSAZ_SCNryJB5/view?usp=sharing"
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ["Healthy", "Early Blight", "Late Blight"]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match model input size
    image = np.expand_dims(image, axis=0)  # Expand dimensions for batch
    return image

# Streamlit UI
st.title("🥔 Potato Disease Detection using AI")
st.markdown("Upload an image of a potato leaf to detect disease.")

# File uploader
uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])

# Predict button
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display the result
        st.subheader("🔎 Prediction Result")
        st.write(f"Detected Disease: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        if predicted_class == "Healthy":
            st.success("✅ The plant is healthy!")
        else:
            st.error(f"⚠ The plant has {predicted_class} disease. Consider treatment.")
