import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/disaster.h5")
categories = ["fire", "flood", "earthquake", "normal"]  # adjust based on your model

def run_disaster_detection(camera):
    ret, frame = camera.read()
    if not ret:
        return "No Input"

    height, width = model.input_shape[1:3]
    img = cv2.resize(frame, (width, height)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return categories[np.argmax(prediction)]