import os
import librosa
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- CONFIGURATION ---
# CONFIGURATION
BASE_DIR = r'C:/Users/NSPatil/OneDrive/Desktop/SEM VI/Sound/Sound' 
TRAIN_DIR = os.path.join(BASE_DIR, 'Training')
VAL_DIR = os.path.join(BASE_DIR, 'Validation')
CLASSES = ['Background', 'Gunshot'] 

def extract_features_advanced(file_path):
    try:
        # Load Audio
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Normalize Volume (Boosts distant/faint sounds)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Mean AND Max Features (80 total)
        mfccs_mean = np.mean(mfccs.T, axis=0) 
        mfccs_max = np.max(mfccs.T, axis=0)   
        
        combined_features = np.hstack([mfccs_mean, mfccs_max])
        return combined_features
        
    except Exception as e:
        return None

def create_dataset(directory):
    features = []
    labels = []
    print(f"\nScanning directory: {directory}")
    
    for label in CLASSES:
        path = os.path.join(directory, label)
        
        if not os.path.exists(path):
            path = os.path.join(directory, label.lower())
            
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.wav', '.mp3'))]
            
            # NO LIMIT: Processing all files
            print(f" -> {label}: Processing all {len(files)} files")

            for file in tqdm(files, desc=f"Processing {label}"):
                file_path = os.path.join(path, file)
                data = extract_features_advanced(file_path)
                
                if data is not None:
                    features.append(data)
                    labels.append(label)
        else:
            print(f"❌ Warning: Folder not found: {path}")

    return np.array(features), np.array(labels)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Processing Training Data (This might take a while for 28,000+ files)...")
    X_train, y_train = create_dataset(TRAIN_DIR)

    print("\n2. Processing Validation Data...")
    X_val, y_val = create_dataset(VAL_DIR)

    print("\n3. Encoding Labels...")
    le = LabelEncoder()
    y_train_encoded = to_categorical(le.fit_transform(y_train))
    y_val_encoded = to_categorical(le.transform(y_val))

    print("\n4. Saving Enhanced Data to Disk...")
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train_encoded)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val_encoded)
    np.save('classes.npy', le.classes_)

    print("\n✅ DONE! All data processed and saved.")
    print(f"   Training Data Shape: {X_train.shape} (Should end in 80)")
    print(f"   Validation Data Shape: {X_val.shape} (Should end in 80)")