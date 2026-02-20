import numpy as np
from PIL import Image
from PytorchWildlife.models import detection as pw_detection

# 1. Initialize the MDv6-Compact model
# It will automatically download the YOLOv9-compact weights (~25MB) on the first run.
print("Loading MegaDetector v6 Compact...")
scout_model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
print("✅ Scout Model loaded successfully!\n")

def analyze_camera_trap(image_path):
    print(f"Analyzing: {image_path}...")
    try:
        # 2. Load the image and convert to RGB array
        img = np.array(Image.open(image_path).convert("RGB"))
        
        # 3. Perform AI Inference
        # This is the lightning-fast YOLOv9 forward pass
        result = scout_model.single_image_detection(img)
        
        detections = result.get('detections')
        
        if len(detections) == 0:
            print("Status: Clear. No animals, humans, or vehicles detected.")
            return

        print(f"🎯 Alert: Found {len(detections)} object(s) in frame!")
        print("-" * 40)
        
        # 4. Parse the results
        # MegaDetector classes: 0 = Animal, 1 = Person, 2 = Vehicle
        class_names = {0: "Animal", 1: "Person", 2: "Vehicle"}
        
        for bbox, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            label = class_names.get(int(class_id), "Unknown")
            
            print(f"Type: {label.upper()}")
            print(f"Confidence: {confidence * 100:.1f}%")
            # The bounding box coordinates (x_min, y_min, x_max, y_max)
            # You will use these coordinates later to crop the image!
            print(f"Location: {bbox}\n")
            
    except Exception as e:
        print(f"Error processing image: {e}")

# --- TEST AREA ---
# Find a test image. A photo from the Western Ghats, a trail camera 
# dataset, or even a picture of yourself holding an object will work.
TEST_IMAGE = "C:/Users/NSPatil/Downloads/ML/eles_close_to_village_ahp_2-790x593.jpg"

analyze_camera_trap(TEST_IMAGE)