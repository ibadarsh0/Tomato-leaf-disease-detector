import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
import os

Model.save("Model/tomato_disease_model.keras")  # or use .h5 if preferred


# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = '../Dataset/train'
VALID_DIR = '../Dataset/validation'
MODEL_PATH = '../Model/tomato_disease_model.keras'

# --- 1. Data Preprocessing & Augmentation ---
# Rescale pixels and apply augmentations to the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale pixels for the validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get the number of classes
num_classes = len(train_generator.class_indices)
print(f"Found {num_classes} classes.")
print(train_generator.class_indices) # Print class names and their indices

# --- 2. Model Building (Transfer Learning with MobileNetV2) ---
# Load the base MobileNetV2 model pre-trained on ImageNet, without the top layer
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Add dropout for regularization
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. Compile and Train the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for saving the best model and stopping early
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

print(f"\nTraining complete. Best model saved to {MODEL_PATH}")