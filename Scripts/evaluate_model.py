import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# --- Configuration ---
MODEL_PATH = '../Model/tomato_disease_model.keras'
TEST_DIR = '../Dataset/test' # Make sure you have a test set
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 1. Load Model and Test Data ---
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# We use shuffle=False to keep data in order for the confusion matrix
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- 2. Evaluate Performance ---
# Get overall accuracy
loss, accuracy = model.evaluate(test_generator)
print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

# Get predictions
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --- 3. Evaluate Efficiency ---
# Model Size
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"\nModel Size: {model_size_mb:.2f} MB")

# Inference Speed
# Load one image to test inference time
sample_image, _ = test_generator.next()
single_image = sample_image[0:1] # Get the first image from the batch

# Run one prediction to "warm up"
model.predict(single_image)

# Time the prediction
start_time = time.time()
model.predict(single_image)
end_time = time.time()
inference_time = (end_time - start_time) * 1000 # in milliseconds
print(f"Inference Speed (on one image): {inference_time:.2f} ms")