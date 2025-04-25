import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Parameters
DATA_PATH = "gunshot_audio"
IMG_SIZE = (128, 128)
LABELS = ['background', 'gunshot']
X, y = [], []

# Convert audio to mel spectrogram image arrays
def extract_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spect = librosa.feature.melspectrogram(y=y, sr=sr)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect = librosa.util.fix_length(spect, size=IMG_SIZE[1], axis=1)
    spect = spect[:IMG_SIZE[0], :]
    spect = (spect - spect.min()) / (spect.max() - spect.min())  # normalize
    return spect

# Load and convert all files
for idx, label in enumerate(labels):
    folder = os.path.join(DATA_PATH, label)
    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            try:
                spec = extract_spectrogram(os.path.join(folder, fname))
                X.append(spec)
                y.append(idx)
            except Exception as e:
                print(f"Skipped {fname}: {e}")

X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
y = np.array(y)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=16)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/gunshot_classifier.h5")
print("✅ Gunshot CNN model trained and saved as .h5")
