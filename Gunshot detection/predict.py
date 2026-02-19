import numpy as np
import librosa
import tensorflow as tf
import os

# 1. Suppress TensorFlow informational logs for a cleaner output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Load Model & Classes
print("Loading model...")
try:
    model = tf.keras.models.load_model('best_gunshot_model.keras')
    classes = np.load('classes.npy')
    print(f"✅ Model ready. Detecting classes: {classes}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure 'best_gunshot_model.keras' and 'classes.npy' are in the same folder.")
    exit()

def predict_sound(file_path):
    print(f"\n🔍 Analyzing: {os.path.basename(file_path)}")
    
    try:
        # A. LOAD AUDIO
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # B. CHECK & NORMALIZE VOLUME (Crucial for distant sounds)
        if len(audio) == 0 or np.max(np.abs(audio)) == 0:
            print("❌ ERROR: The audio file is empty or completely silent.")
            return
            
        audio = audio / np.max(np.abs(audio))
        
        # C. EXTRACT FEATURES (Mean + Max = 80 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        mfccs_mean = np.mean(mfccs.T, axis=0) # Background context
        mfccs_max = np.max(mfccs.T, axis=0)   # Sharp spikes (gunshots)
        
        # Combine into exactly 80 features
        combined_features = np.hstack([mfccs_mean, mfccs_max])
        mfccs_processed = combined_features.reshape(1, -1)
        
        # D. PREDICT
        prediction = model.predict(mfccs_processed, verbose=0)
        
        class_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][class_index]
        label = classes[class_index]
        
        # E. DISPLAY RESULTS
        print("-" * 35)
        print(f"🎯 PREDICTION: {label.upper()}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        print("-" * 35)
        
        # Show raw probabilities so you can see exactly how the model is "thinking"
        print(f"Raw Probabilities: {classes[0]} = {prediction[0][0]:.4f} | {classes[1]} = {prediction[0][1]:.4f}")

    except Exception as e:
        print(f"Error processing file: {e}")

# --- TEST AREA ---
# UPDATE THIS with the path to a gunshot or background file you want to test
TEST_FILE = "C:/Users/NSPatil/OneDrive/Desktop/SEM VI/mp/Project code files and datasets/Sound/Validation/Gunshot/5B0AFA51_4e935b106-d5af-4aac-a0bd-0ca416633275ec86621c-c7d8-4c03-9f7b-21e3e7e457006e0ebe11-7610-465b-9485-8b38209c0f59.WAV"

if os.path.exists(TEST_FILE):
    predict_sound(TEST_FILE)
else:
    print(f"\n⚠️ Please update the TEST_FILE variable. Could not find: {TEST_FILE}")