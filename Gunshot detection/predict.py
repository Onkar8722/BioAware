import numpy as np
import librosa
import tensorflow as tf
import os

# 1. Load Model & Classes
print("Loading model...")
model = tf.keras.models.load_model('best_gunshot_model.keras')
classes = np.load('classes.npy')

# CONFIGURATION
CHUNK_SIZE = 2.0  # Analyze 2 seconds at a time
OVERLAP = 0.5     # 50% overlap to catch shots happening between chunks

def predict_smart_scan(file_path):
    print(f"\nScanning: {os.path.basename(file_path)}...")
    
    try:
        # Load the FULL audio file
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"Audio Duration: {duration:.2f} seconds")
        
        # Calculate samples per chunk
        # sr = samples per second (usually 22050)
        chunk_samples = int(CHUNK_SIZE * sr)
        step_samples = int(chunk_samples * (1 - OVERLAP))
        
        # If file is shorter than chunk, pad it
        if len(audio) < chunk_samples:
             audio = np.pad(audio, (0, chunk_samples - len(audio)))
        
        highest_confidence = 0.0
        found_gunshot = False
        
        # --- THE SCANNING LOOP ---
        # We slide a window across the audio file
        for i in range(0, len(audio) - chunk_samples + 1, step_samples):
            # 1. Cut out the chunk
            chunk = audio[i : i + chunk_samples]
            
            # 2. Extract Features (Same as training)
            mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            mfccs_processed = mfccs_scaled.reshape(1, -1)
            
            # 3. Predict
            prediction = model.predict(mfccs_processed, verbose=0)
            class_index = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][class_index]
            label = classes[class_index]
            
            # Track the results
            timestamp = i / sr
            print(f"  Time {timestamp:.1f}s - {timestamp+CHUNK_SIZE:.1f}s: {label} ({confidence*100:.1f}%)")
            
            if label == 'Gunshot':
                if confidence > highest_confidence:
                    highest_confidence = confidence
                # If we are very sure, we can stop early or flag it
                if confidence > 0.80: 
                    found_gunshot = True
        
        # --- FINAL VERDICT ---
        print("-" * 30)
        if found_gunshot:
            print(f"🚨 ALERT: GUNSHOT DETECTED!")
            print(f"   Max Confidence: {highest_confidence*100:.2f}%")
            return "Gunshot"
        else:
            print(f"✅ Status: Normal (Background Noise)")
            return "Background"

    except Exception as e:
        print(f"Error: {e}")
        return None

# --- TEST AREA ---
# Use the SAME file that failed before
TEST_FILE ="C:/Users/NSPatil/Downloads/freesound_community-gun-shot-1-7069.mp3"

# Run the smart scan
predict_smart_scan(TEST_FILE)