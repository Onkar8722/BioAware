import numpy as np
import librosa
import time
import os

# NOTE: For testing on your PC, we use the standard tensorflow library.
# When you move to a Raspberry Pi, change this line to:
# from tflite_runtime.interpreter import Interpreter
import tensorflow as tf 

# --- CONFIGURATION ---
TFLITE_MODEL_PATH = 'gunshot_model.tflite'
CLASSES_PATH = 'classes.npy'

def fast_predict(file_path):
    print(f"\n⚡ TFLite Fast Prediction ⚡")
    print(f"Analyzing: {os.path.basename(file_path)}")
    
    # Start the overall timer
    start_time = time.time()
    
    try:
        # 1. Initialize TFLite Interpreter
        # This loads the tiny model structure into memory instantly
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        classes = np.load(CLASSES_PATH)

        # 2. Load and Process Audio 
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Normalize
        if len(audio) == 0 or np.max(np.abs(audio)) == 0:
             print("❌ Audio empty.")
             return
        audio = audio / np.max(np.abs(audio))
        
        # Extract 80 features (Mean + Max)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_max = np.max(mfccs.T, axis=0)
        
        # Combine and explicitly convert to float32 (TFLite strictly requires float32)
        combined_features = np.hstack([mfccs_mean, mfccs_max])
        input_data = np.array(combined_features.reshape(1, -1), dtype=np.float32)

        # 3. Perform AI Inference (The actual "brain" work)
        inference_start = time.time()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        inference_end = time.time()

        # 4. Decode Results
        class_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][class_index]
        label = classes[class_index]
        
        total_time = time.time() - start_time
        inference_time = inference_end - inference_start
        
        # 5. Display
        print("-" * 35)
        print(f"🎯 RESULT: {label.upper()}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        print("-" * 35)
        print(f"⏱️ Pure AI Inference Time: {inference_time*1000:.2f} ms")
        print(f"⏱️ Total Time (incl. audio load): {total_time:.2f} seconds")

    except Exception as e:
         print(f"Error: {e}")

# --- TEST AREA ---
# UPDATE THIS with a path to one of your test audio files
TEST_FILE = "C:/Users/NSPatil/OneDrive/Desktop/SEM VI/mp/Project code files and datasets/Sound/Validation/Gunshot/5B0AFA51_4e935b106-d5af-4aac-a0bd-0ca416633275ec86621c-c7d8-4c03-9f7b-21e3e7e457006e0ebe11-7610-465b-9485-8b38209c0f59.WAV"


if os.path.exists(TEST_FILE):
    fast_predict(TEST_FILE)
else:
    print(f"\n⚠️ Please update the TEST_FILE variable. Could not find: {TEST_FILE}")