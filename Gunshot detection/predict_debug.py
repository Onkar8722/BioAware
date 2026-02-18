import numpy as np
import librosa
import tensorflow as tf
import os

# 1. Load Model
print("Loading model...")
model = tf.keras.models.load_model('best_gunshot_model.keras')
classes = np.load('classes.npy')

def predict_with_fix(file_path):
    print(f"\n🔍 DIAGNOSING: {os.path.basename(file_path)}")
    
    try:
        # A. LOAD AUDIO
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # CHECK 1: Is the file actually empty/silent?
        if np.max(np.abs(audio)) == 0:
            print("❌ ERROR: The audio file is empty (All Zeros)!")
            print("   Fix: Your computer might lack the codec to read this MP3.")
            print("   Try converting it to WAV manually or use a different file.")
            return

        # CHECK 2: Volume Level
        max_vol = np.max(np.abs(audio))
        print(f"   Original Volume (Peak): {max_vol:.4f}")
        
        # B. FIX: NORMALIZE AUDIO (Make it Loud!)
        # This scales the audio so the loudest point is always 1.0 (Maximum loudness)
        # This helps if your training data was loud but this file is quiet.
        audio_normalized = audio / max_vol
        print(f"   ✅ Applied Volume Normalization (Scale 1.0)")

        # C. EXTRACT FEATURES
        mfccs = librosa.feature.mfcc(y=audio_normalized, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        mfccs_processed = mfccs_scaled.reshape(1, -1)
        
        # D. PREDICT
        prediction = model.predict(mfccs_processed, verbose=0)
        class_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][class_index]
        label = classes[class_index]
        
        print("-" * 30)
        print(f"PREDICTION: {label.upper()}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("-" * 30)
        print(f"Raw Probabilities: {classes[0]}={prediction[0][0]:.4f}, {classes[1]}={prediction[0][1]:.4f}")

    except Exception as e:
        print(f"Error: {e}")

# --- TEST AREA ---
TEST_FILE = "C:/Users/NSPatil/OneDrive/Desktop/SEM VI/Sound/Sound/Validation/Gunshot/5B0AFA4D_3469ea810-1ff7-4e52-9234-622cb952b026a2d8aadf-961d-4954-b29f-b09b2ca5ff8161423544-5107-43e2-aa29-cb9a12c7c2ed.WAV"

predict_with_fix(TEST_FILE)