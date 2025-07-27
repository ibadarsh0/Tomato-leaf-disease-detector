import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
# Adjust the path to your trained model
MODEL_PATH = '../Model/tomato_disease_model.keras'
IMG_SIZE = (224, 224)

# Define the class names in the correct order (based on your training output)
CLASS_NAMES = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
] # Make sure this order matches your training generator's class_indices

# --- 1. Load the Trained Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle error, maybe show a message to the user
    exit()

# --- 2. GUI Application ---
class DiseaseDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tomato Leaf Disease Detector 🍅")
        self.geometry("600x600")

        # Style
        self.style = ttk.Style(self)
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 14))
        self.style.configure('Result.TLabel', font=('Helvetica', 16, 'bold'))

        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Image display
        self.image_label = ttk.Label(main_frame, text="Upload an image to start", style='TLabel')
        self.image_label.pack(pady=20)

        # Upload button
        upload_button = ttk.Button(main_frame, text="Upload Image", command=self.upload_and_predict, style='TButton')
        upload_button.pack(pady=10)

        # Result display
        self.result_label = ttk.Label(main_frame, text="", style='Result.TLabel', wraplength=500, justify=tk.CENTER)
        self.result_label.pack(pady=20)

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        # Display the uploaded image
        img = Image.open(file_path)
        img.thumbnail((300, 300)) # Resize for display
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

        # Preprocess the image for the model
        processed_image = self.preprocess_image(file_path)
        
        # Make prediction
        prediction = model.predict(processed_image)
        confidence = np.max(prediction[0]) * 100
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index].replace('_', ' ')

        # Update the result label
        result_text = f"Prediction: {predicted_class_name}\nConfidence: {confidence:.2f}%"
        self.result_label.config(text=result_text)

    def preprocess_image(self, file_path):
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Create a batch
        img_array /= 255.0 # Rescale
        return img_array

if __name__ == "__main__":
    app = DiseaseDetectorApp()
    app.mainloop()