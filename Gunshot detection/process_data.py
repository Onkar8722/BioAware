import os
import librosa
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
# Data path is: 'C:/Users/NSPatil/OneDrive/Desktop/SEM VI/Sound/Sound'
BASE_DIR = r'C:/Users/NSPatil/OneDrive/Desktop/SEM VI/Sound/Sound' 

TRAIN_DIR = os.path.join(BASE_DIR, 'Training')
VAL_DIR = os.path.join(BASE_DIR, 'Validation')

# specific folder names inside Training/Validation
CLASSES = ['Background', 'Gunshot'] 

def extract_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract MFCCs (The "Fingerprint")
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Take the average to get a 1D array of 40 numbers
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_dataset(directory):
    features = []
    labels = []
    
    print(f"Scanning directory: {directory}")
    
    for label in CLASSES:
        path = os.path.join(directory, label)
        
        # Check if folder exists
        if not os.path.exists(path):
            print(f"WARNING: Folder not found: {path}")
            # Try lowercase just in case (windows is usually case-insensitive but python can be strict)
            path = os.path.join(directory, label.lower())
            if os.path.exists(path):
                print(f"Found lowercase folder instead: {path}")
            else:
                continue

        print(f" -> Processing class '{label}'...")
        
        files = [f for f in os.listdir(path) if f.lower().endswith('.wav')]
        total_files = len(files)
        
        for i, file in enumerate(files):
            file_path = os.path.join(path, file)
            data = extract_features(file_path)
            
            if data is not None:
                features.append(data)
                labels.append(label)
                
            # Print progress every 50 files
            if (i+1) % 50 == 0:
                print(f"    Processed {i+1}/{total_files}")

    return np.array(features), np.array(labels)

# --- MAIN EXECUTION ---

print("1. Processing Training Data...")
X_train, y_train = create_dataset(TRAIN_DIR)

print("\n2. Processing Validation Data...")
X_val, y_val = create_dataset(VAL_DIR)

# 3. Encode Labels (Background -> 0, Gunshot -> 1)
print("\n3. Encoding Labels...")
le = LabelEncoder()
y_train_encoded = to_categorical(le.fit_transform(y_train))
y_val_encoded = to_categorical(le.transform(y_val))

# 4. Save to disk (So we don't have to do this again!)
print("\n4. Saving processed data to disk...")
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train_encoded)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val_encoded)
np.save('classes.npy', le.classes_)

print("\nDONE! Data saved successfully.")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Validation Samples: {X_val.shape[0]}")