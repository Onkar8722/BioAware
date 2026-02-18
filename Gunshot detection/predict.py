import numpy as np
import librosa
import tensorflow as tf
import os

# 1. Ignore the "oneDNN" warnings (They are just performance logs, not errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Load Model & Classes
print("Loading model...")
model = tf.keras.models.load_model('best_gunshot_model.keras')
classes = np.load('classes.npy')
print(f"Model ready. Classes: {classes}")

def predict_single_file(file_path):
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    
    try:
        # 1. Load audio (EXACTLY like training)
        # librosa loads it and resamples to 22050Hz automatically
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        print(f"  Duration: {len(audio)/sample_rate:.2f}s | Sample Rate: {sample_rate}Hz")
        
        # 2. Extract Features
        # Note: We do NOT pad with zeros here. We take the mean of whatever we have.
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # 3. Reshape (1 sample, 40 features)
        mfccs_processed = mfccs_scaled.reshape(1, -1)
        
        # 4. Predict
        prediction = model.predict(mfccs_processed, verbose=0)
        
        # 5. Decode Result
        class_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][class_index]
        label = classes[class_index]
        
        # Print
        print("-" * 30)
        print(f"PREDICTION: {label.upper()}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("-" * 30)
        
        # Full probabilities (Debug info)
        print(f"Raw Probabilities: {classes[0]}={prediction[0][0]:.4f}, {classes[1]}={prediction[0][1]:.4f}")

    except Exception as e:
        print(f"Error: {e}")

# --- TEST AREA ---
# Replace with your MP3 file path
TEST_FILE = "C:/Users/NSPatil/OneDrive/Desktop/SEM VI/Sound/Sound/Validation/Gunshot/5B0AFA4D_3469ea810-1ff7-4e52-9234-622cb952b026a2d8aadf-961d-4954-b29f-b09b2ca5ff8161423544-5107-43e2-aa29-cb9a12c7c2ed.WAV"

predict_single_file(TEST_FILE)