from ultralytics import YOLO
import cv2
import os
import glob

# --- CONFIGURATION ---
# 1. Path to your newly trained weights
# (I updated this to 'weapon_sniper2' based on your logs)
MODEL_PATH = r"C:/Users/NSPatil/OneDrive/Desktop/My Projects/AI for Wildlife/runs/detect/weapon_sniper2/weights/best.pt"

# 2. Folder containing the cropped images from Stage 1
INPUT_FOLDER = "sniper_crops"

# 3. Where to save the final proof (images with weapon boxes drawn)
OUTPUT_FOLDER = "final_detections"

# Confidence threshold (only show weapons if 40% sure)
CONF_THRESHOLD = 0.40

def scan_for_weapons():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not find model weights at: {MODEL_PATH}")
        return

    print("🚀 Loading the Weapon Hunter Model...")
    model = YOLO(MODEL_PATH)
    
    # Get all images from the input folder
    image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + glob.glob(os.path.join(INPUT_FOLDER, "*.png"))
    
    if not image_files:
        print(f"⚠️ No images found in '{INPUT_FOLDER}'. Run the 'sniper_crop.py' script first!")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"📂 Scanning {len(image_files)} potential targets in '{INPUT_FOLDER}'...\n")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # Run Inference
        # save=True will automatically draw the boxes and save to 'runs/detect/predict'
        # But we will do it manually to have full control over the output folder
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        
        result = results[0] # We are only processing one image at a time
        
        if len(result.boxes) > 0:
            print(f"🚨 WEAPON DETECTED in {filename}!")
            
            # Draw the boxes on the image
            # 'render()' isn't in YOLOv8/11, so we use plot()
            annotated_frame = result.plot()
            
            # Save to our custom folder
            save_path = os.path.join(OUTPUT_FOLDER, "DETECTED_" + filename)
            cv2.imwrite(save_path, annotated_frame)
            
            # Print details
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                conf = float(box.conf[0])
                print(f"   -> Found: {class_name} ({conf*100:.1f}%)")
            print("-" * 30)
            
        else:
            print(f"✅ Clear: No weapons found in {filename}.")

    print(f"\n🎯 Job Complete. Check the '{OUTPUT_FOLDER}' folder for evidence.")

if __name__ == "__main__":
    scan_for_weapons()