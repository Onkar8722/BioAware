import tensorflow as tf

model = tf.keras.models.load_model("models/best_model.h5", compile=False)
print("Expected model input shape:", model.input_shape)

