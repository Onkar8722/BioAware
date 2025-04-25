import sounddevice as sd
import numpy as np
import librosa
import joblib

# Load the joblib model trained on MFCC features
model = joblib.load("models/gunshot_model.pkl")


DURATION = 2  # seconds
SAMPLE_RATE = 22050


def run_gunshot_detection():
    try:
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio.flatten(), sr=SAMPLE_RATE, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

        # Make prediction
        prediction = model.predict(mfccs_mean)
        return "Gunshot" if prediction[0] == 1 else "No Gunshot"
    except Exception as e:
        return f"Error: {str(e)}"
    

