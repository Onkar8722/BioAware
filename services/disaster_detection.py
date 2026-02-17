import cv2
import tensorflow as tf
import numpy as np

# Load the model (safely, in case it was saved with training configs)
model = tf.keras.models.load_model("models/disaster.h5", compile=False)

# Define your category labels
categories = ["Cyclone", "Fire", "Flood", "Earthquake", "normal"]  # adjust if needed

def run_disaster_detection(camera):
    ret, frame = camera.read()
    if not ret:
        return "No Input"

    # Get expected input size from model
    input_shape = model.input_shape  # e.g., (None, 180, 180, 3)
    height, width = input_shape[1:3]

    # Resize and preprocess image
    img = cv2.resize(frame, (width, height))
    img = img.astype('float32') / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # shape: (1, height, width, 3)

    # Predict
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    return categories[predicted_index]
