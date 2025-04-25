import cv2
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model("models/animal_classifier.h5")
#categories = ["cat", "dog", "snake", "human"]  # Update with your actual class list
my_file = open("C:/Users/NSPatil/OneDrive/Desktop/SEM VI/mp/Project code files and datasets/archive (8)/name of the animals.txt")
data = my_file.read()
categories =data.replace("\n", " ").split(" ")
my_file.close()
# Removed internal camera access so it can be shared from app.py
def run_animal_detection(camera):
    camera.grab()
    ret, frame = camera.retrieve()
    if not ret:
        return "No Input"

    height, width = model.input_shape[1:3]
    img = cv2.resize(frame, (width, height)) / 255.0
    img = img.reshape(1, height, width, 3)
    prediction = model.predict(img)
    return categories[np.argmax(prediction)]

def predict_animal_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        height, width = model.input_shape[1:3]
        img = cv2.resize(img, (width, height)) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        return categories[np.argmax(prediction)]
    except Exception as e:
        return f"Error: {str(e)}"