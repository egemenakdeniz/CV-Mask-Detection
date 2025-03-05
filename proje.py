import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# Dataset Preparation
DATASET_PATH = "dataset"
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations.csv")
MASKED_DIR = os.path.join(DATASET_PATH, "with_mask")
NON_MASKED_DIR = os.path.join(DATASET_PATH, "without_mask")

# Organize dataset if not already organized
if not os.path.exists(MASKED_DIR) or not os.path.exists(NON_MASKED_DIR):
    os.makedirs(MASKED_DIR, exist_ok=True)
    os.makedirs(NON_MASKED_DIR, exist_ok=True)

    # Read the CSV file
    annotations = pd.read_csv(ANNOTATIONS_PATH)

    for _, row in annotations.iterrows():
        img_name = row['filename']
        label = row['class']
        src = os.path.join(IMAGE_PATH, img_name)
        
        if label == "mask":
            dst = os.path.join(MASKED_DIR, img_name)
        else:
            dst = os.path.join(NON_MASKED_DIR, img_name)
        
        if os.path.exists(src):  # Move the file if it exists
            copyfile(src, dst)

print("Dataset organized successfully.")

# Model Architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load or Train the Model
MODEL_PATH = "mask_detection_model.h5"

try:
    model = load_model(MODEL_PATH)
    print("Pre-trained model loaded successfully.")
except:
    print("No pre-trained model found. Training a new one...")
    model = create_model()

    # Data Augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH, target_size=(224, 224), batch_size=32, class_mode='binary', subset='training')

    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH, target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation')

    model.fit(train_generator, validation_data=val_generator, epochs=10)
    model.save(MODEL_PATH)
    print("Model trained and saved successfully.")

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))  # Adjust to model input size
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize pixel values

        prediction = model.predict(face)[0]
        masked = prediction[0] > 0.5  # Classification threshold

        label = "Wearing Mask" if masked else "No Mask"
        color = (0, 255, 0) if masked else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
