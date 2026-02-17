import numpy as np
np.complex = complex  # Fix for deprecated np.complex used by librosa

import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import io
import sounddevice as sd
import joblib
import tensorflow as tf

# Load the Keras model from .pkl
model = joblib.load("C:/Users/NSPatil/OneDrive/Desktop/AI for Wildlife/models/gunshot_model.pkl")  # Replace with full path if needed

# Parameters
SAMPLE_RATE = 22050
DURATION = 2  # seconds of recording
IMG_SIZE = 128

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return np.squeeze(audio)

def audio_to_spectrogram_image(audio, sr=SAMPLE_RATE, img_size=IMG_SIZE):
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    fig = plt.figure(figsize=(2, 2), dpi=64)
    plt.axis('off')
    librosa.display.specshow(S_DB, sr=sr, cmap='gray')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype('float32') / 255.0
    return image.reshape(1, img_size, img_size, 1)

def run_gunshot_detection():
    audio = record_audio()
    image_input = audio_to_spectrogram_image(audio)
    prediction = model.predict(image_input)[0]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"\n🔍 Prediction: {'Gunshot' if predicted_class == 1 else 'No Gunshot'} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    run_gunshot_detection()
