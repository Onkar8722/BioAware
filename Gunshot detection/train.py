import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# 1. Load Data
print("Loading data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

print(f"Input Shape: {X_train.shape} (The second number should be 80)")

# 2. Compute Class Weights (The safety net for imbalance)
# Convert one-hot encoded labels back to 0s and 1s for the sklearn function
y_integers = np.argmax(y_train, axis=1)

# Calculate weights to penalize the model heavily if it misses a gunshot
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_integers), 
    y=y_integers
)
class_weights_dict = dict(enumerate(class_weights))
print(f"⚖️ Computed Class Weights: {class_weights_dict}")

# 3. Build the Model Architecture
model = Sequential()

# Input Layer (80 features from Mean + Max)
model.add(Dense(256, input_shape=(80,)))
model.add(BatchNormalization()) # Stabilizes learning for distant/low-volume audio
model.add(Activation('relu'))
model.add(Dropout(0.4))

# Hidden Layer 2
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

# Hidden Layer 3
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))

# Output Layer (2 Classes: Background, Gunshot)
model.add(Dense(2))
model.add(Activation('softmax'))

# 4. Compile
model.compile(loss='categorical_crossentropy', 
              metrics=['accuracy'], 
              optimizer='adam')

# 5. Callbacks
checkpoint = ModelCheckpoint('best_gunshot_model.keras', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             verbose=1)

# Gradually reduce learning rate to fine-tune the model when it stops improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=5, 
                              min_lr=0.001,
                              verbose=1)

# 6. Train the Model
print("\nStarting Training on Enhanced & Balanced Data...")
history = model.fit(X_train, y_train, 
          batch_size=32, 
          epochs=100, 
          validation_data=(X_val, y_val), 
          class_weight=class_weights_dict,  # Applying the weights here!
          callbacks=[checkpoint, reduce_lr], 
          verbose=1)

#plotting training history
import matplotlib.pyplot as plt 
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


print("\nTraining finished! Best model saved as 'best_gunshot_model.keras'")

