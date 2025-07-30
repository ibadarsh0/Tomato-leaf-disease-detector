import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model (adjust path if needed)
model = tf.keras.models.load_model("Model/tomato_disease_model.keras")

# Define class names
class_names = ['Healthy', 'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Spider Mites']

# Prediction function
def predict(img):
    img = img.resize((224, 224))  # Resize to model's input shape
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]
    pred = model.predict(img)
    return class_names[np.argmax(pred)]

# Streamlit UI
st.title("🍅 Tomato Leaf Disease Detection")
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    result = predict(img)
    st.success(f"Prediction: *{result}*")