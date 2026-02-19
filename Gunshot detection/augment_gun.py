import librosa
import soundfile as sf
import numpy as np
import os
from tqdm import tqdm

# --- FIXED PATHS (Using raw strings 'r' to handle Windows backslashes) ---
INPUT_DIR = r'C:/Users/NSPatil/OneDrive/Desktop/SEM VI/mp/Project code files and datasets/Sound/Training/Gunshot' 
OUTPUT_DIR = r'C:/Users/NSPatil/OneDrive/Desktop/SEM VI/Sound/Sound/Validation/Gunshot'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Generating augmented data from: {INPUT_DIR}")

# --- THE FIX: Case-insensitive check for both .wav and .mp3 ---
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.wav', '.mp3','.WAV','.MP3'))]

if len(files) == 0:
    print("❌ ERROR: Still found 0 files. Please check if your INPUT_DIR path is perfectly correct!")
else:
    print(f"✅ Found {len(files)} files. Starting augmentation...")

for file in tqdm(files):
    try:
        file_path = os.path.join(INPUT_DIR, file)
        y, sr = librosa.load(file_path, res_type='kaiser_fast')
        
        # Remove the old extension so we can standardize everything to .wav
        filename = os.path.splitext(file)[0] 
        
        # Save Original 
        sf.write(os.path.join(OUTPUT_DIR, f"{filename}.wav"), y, sr)
        
        # Variation 1: Pitch Shift (Higher tone)
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        sf.write(os.path.join(OUTPUT_DIR, f"{filename}_p2.wav"), y_pitch, sr)
        
        # Variation 2: Time Stretch (Faster)
        y_fast = librosa.effects.time_stretch(y, rate=1.1)
        sf.write(os.path.join(OUTPUT_DIR, f"{filename}_fast.wav"), y_fast, sr)
        
        # Variation 3: Time Stretch (Slower)
        y_slow = librosa.effects.time_stretch(y, rate=0.85)
        sf.write(os.path.join(OUTPUT_DIR, f"{filename}_slow.wav"), y_slow, sr)
        
        # Variation 4: Add Noise (Simulate wind/rain)
        noise = np.random.randn(len(y))
        y_noise = y + 0.005 * noise
        sf.write(os.path.join(OUTPUT_DIR, f"{filename}_noise.wav"), y_noise, sr)

    except Exception as e:
        print(f"Skipping {file}: {e}")

print(f"\n✅ Done! You now have ~{len(files)*5} gunshot samples in '{OUTPUT_DIR}'")