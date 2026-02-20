from ultralytics import YOLO
import os

# --- CONFIGURATION ---
# UPDATE THIS PATH to point to the data.yaml file inside your extracted Roboflow dataset folder
# Use the raw string 'r' to ensure Windows backslashes are read correctly
DATA_YAML_PATH = r"C:/Users/NSPatil/OneDrive/Desktop/SEM VI/Sound/Weapon Detection/data.yaml"

def train_weapon_detector():
    # Safety check
    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ ERROR: Cannot find data.yaml at {DATA_YAML_PATH}")
        print("Please check the path and try again.")
        return

    print("🚀 Initializing YOLO11-Nano (Edge Optimized)...")
    # This automatically downloads the tiny foundation model weights
    model = YOLO("yolo11n.pt")
    
    print("🧠 Starting Stage 2 Training...")
    
    # The Training Pipeline
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=50,             # 50 epochs is a solid baseline to see if it learns without overfitting
        imgsz=320,             # PRO TIP: We use 320 instead of 640 because your Stage 1 crops are already small. This doubles training speed!
        batch=16,              # Process 16 images at a time (lower to 8 if your laptop throws a memory error)
        device="cpu",          # Using CPU. Change to "0" if you have an NVIDIA GPU with CUDA installed
        patience=10,           # Early stopping: halts training early if accuracy stops improving for 10 epochs
        name="weapon_sniper",  # Saves your specific run results under this folder name
        workers=2              # Keeps CPU usage stable so your laptop doesn't freeze
    )
    
    print("\n✅ Training Complete!")
    print("Your production-ready weights are saved at: runs/detect/weapon_sniper/weights/best.pt")

if __name__ == "__main__":
    train_weapon_detector()