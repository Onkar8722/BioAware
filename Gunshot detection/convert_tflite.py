import tensorflow as tf

print("Loading the heavy Keras model...")
model = tf.keras.models.load_model('best_gunshot_model.keras')

print("Converting to TensorFlow Lite...")
# Initialize the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Add optimizations to make it even smaller/faster
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Perform the conversion
tflite_model = converter.convert()

# Save the new lightweight model
with open('gunshot_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Success! Your model is now saved as 'gunshot_model.tflite'")