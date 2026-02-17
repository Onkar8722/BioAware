import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

# 1. Load the processed data (The ".npy" files we just made)
print("Loading data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

print(f"Training Data Shape: {X_train.shape}")
print(f"Validation Data Shape: {X_val.shape}")

# 2. Build the Model Architecture
# We use a 'Sequential' model, which is a linear stack of layers.
model = Sequential()

# -- Input Layer & Hidden Layer 1 --
# 'Dense' means every neuron is connected to every neuron in the next layer.
# 256 neurons: A good starting size.
# input_shape=(40,): Matches our 40 MFCC features.
# activation='relu': The standard activation function for hidden layers.
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5)) # Dropout ignores 50% of neurons randomly to prevent overfitting.

# -- Hidden Layer 2 --
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# -- Hidden Layer 3 --
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# -- Output Layer --
# 2 neurons: One for 'Background', one for 'Gunshot'.
# activation='softmax': Converts the output into probabilities (e.g., [0.1, 0.9]).
model.add(Dense(2))
model.add(Activation('softmax'))

# 3. Compile the Model
# optimizer='adam': The algorithm that updates the weights (the "learner").
# loss='categorical_crossentropy': The standard loss function for multi-class classification.
model.compile(loss='categorical_crossentropy', 
              metrics=['accuracy'], 
              optimizer='adam')

# Print the model structure
model.summary()

# 4. Train the Model
print("\nStarting training...")

# Checkpoint: Save the best version of the model during training
checkpoint = ModelCheckpoint('best_gunshot_model.keras', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             verbose=1)

history = model.fit(X_train, y_train, 
          batch_size=32,          # Update weights after every 32 samples
          epochs=50,              # Go through the entire dataset 50 times
          validation_data=(X_val, y_val), 
          callbacks=[checkpoint], # Save the best model automatically
          verbose=1)

# 5. Plot the Results (Visualizing Learning)
# This helps us see if the model is learning or just memorizing.
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

print("Training finished. Best model saved as 'best_gunshot_model.keras'")