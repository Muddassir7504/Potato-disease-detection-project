import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import base64


# Set page configuration
st.set_page_config(page_title="Potato Leaf Disease Detector", layout="wide")

# Load the trained model
file_id = "1GB5G1EKz2WqJFSDGmgUAAFEtXeNJ1dXQ"
url = "https://drive.google.com/file/d/1GB5G1EKz2WqJFSDGmgUAAFEtXeNJ1dXQ/view?usp=sharing"
model_path = "trained_plant_disease_model1.keras"



if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path,quiet=False)

model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']


# Custom CSS for Background Image & Styling
st.markdown("""
    <style>
            
        body {
            background-image: url("https://img.freepik.com/premium-photo/leaf-plant-pattern-illustration_728905-2516.jpg");
            background-size: cover; 
            background-repeat: no-repeat;
        }
        }
        
        /* Title Styling */
        .title {
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
            color: #fff;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.6);
        }

        /* Subtitle */
        .subtitle {
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
            color: #f0f0f0;
            text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.5);
        }

        /* File uploader */
        .stFileUploader {
            border: 2px dashed #795548;
            padding: 12px;
            border-radius: 12px;
            text-align: center;
            background: rgba(255, 255, 255, 0.4);
        }

        /* Image container */
        .image-container {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }

        /* Prediction box */
        .prediction-box {
            background: rgba(255, 248, 225, 0.9);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border-left: 6px solid #A67C52;
        }

        /* Confidence score */
        .confidence {
            font-weight: bold;
            font-size: 22px;
            color: #795548;
        }
    </style>
""", unsafe_allow_html=True)




# App Title & Subtitle
st.markdown('<h1 class="title">🥔 Potato Leaf Disease Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image of a potato leaf to detect any disease.</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.markdown('<div class="image-container uploaded-img">', unsafe_allow_html=True)
    st.image(image, caption="📸 Uploaded Image", use_column_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # Ensure image is in RGB mode
    image = image.convert("RGB")

    # Preprocess image
    image = image.resize((128, 128))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Loading animation while making prediction
    with st.spinner("⏳ Analyzing..."):
        predictions = model.predict(image_array)

    predicted_class = np.argmax(predictions, axis=1)[0]  # Get highest probability class
    confidence = np.max(predictions)  # Get confidence score

    # Prediction display box
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction")
    st.write(f"<p class='confidence'>Predicted Class: {class_labels[predicted_class]}</p>", unsafe_allow_html=True)
    st.write(f"*Confidence:* {confidence:.2f}")

    # Show messages based on prediction
    if class_labels[predicted_class] == 'Potato__Early_blight':
        st.warning("⚠ *Early Blight detected!* Use fungicides and improve field management.")
    elif class_labels[predicted_class] == 'Potato_Late_blight':
        st.error("🚨 *Late Blight detected!* Immediate action needed to prevent crop loss.")
    else:
        st.success("✅ *The potato leaf is healthy!* No disease detected.")

    st.markdown("</div>", unsafe_allow_html=True)
