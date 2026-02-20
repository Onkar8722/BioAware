import numpy as np
import cv2
import os
from PIL import Image
from PytorchWildlife.models import detection as pw_detection

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.35  # Ignore anything below 40% certainty
TARGET_CLASS = 1             # In MegaDetector, 1 = Person
OUTPUT_DIR = "sniper_crops"  # Folder to save the cropped humans

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading MDv6-Compact (Stage 1 Scout)...")
scout_model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
print("✅ Scout Model loaded.")

def process_and_crop(image_path):
    print(f"\n🔍 Scanning: {os.path.basename(image_path)}")
    
    try:
        # 1. Load the original image using OpenCV (great for slicing and saving)
        # OpenCV loads images as NumPy arrays, making cropping very fast.
        original_img = cv2.imread(image_path)
        if original_img is None:
            print("❌ Error: Could not load image.")
            return

        # Convert to RGB for MegaDetector (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 2. Stage 1: The Fast Forward Pass
        result = scout_model.single_image_detection(img_rgb)
        detections = result.get('detections')
        
        if len(detections) == 0:
            print("Status: Clear.")
            return

        crop_count = 0

        # 3. Filter, Coordinate Extraction, and Cropping
        for bbox, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            
            # Check 1: Is it a person?
            # Check 2: Is the AI confident enough?
            if int(class_id) == TARGET_CLASS and confidence > CONFIDENCE_THRESHOLD:
                
                # Extract the exact pixel coordinates
                # We convert them to integers because you can't have half a pixel.
                x_min, y_min, x_max, y_max = map(int, bbox)
                
                # Use NumPy array slicing to crop the image. 
                # The format is image[y_start:y_end, x_start:x_end].
                cropped_person = original_img[y_min:y_max, x_min:x_max]
                
                # Save the cropped image for the Weapon Detection model
                crop_filename = os.path.join(OUTPUT_DIR, f"target_{crop_count}_{confidence*100:.0f}conf.jpg")
                cv2.imwrite(crop_filename, cropped_person)
                
                print(f"🚨 CONFIRMED HUMAN: {confidence*100:.1f}% confidence.")
                print(f"   -> Cropped target saved as: {crop_filename}")
                crop_count += 1
                
        if crop_count == 0:
            print("Status: No highly confident humans detected. Ignoring noise.")

    except Exception as e:
        print(f"Error processing image: {e}")

# --- TEST AREA ---
# Update this with the path to your lion/jeep image
TEST_IMAGE = "C:/Users/NSPatil/Downloads/ML/eles_close_to_village_ahp_2-790x593.jpg" 

process_and_crop(TEST_IMAGE)