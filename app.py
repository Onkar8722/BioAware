from flask import Flask, render_template, jsonify, request, Response
from threading import Thread
import os
import cv2
import time
import numpy as np
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')

from services.animal_detection import predict_animal_from_image, run_animal_detection, model as animal_model, categories as animal_categories
from services.disaster_detection import run_disaster_detection
from services.gunshot_detection import run_gunshot_detection
from utils.threat_classifier import classify_threat
from alerts.notifier import send_telegram_alert

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define camera settings directly
def initialize_camera():
    print("Initializing camera with explicit settings...")
    
    # Try multiple camera backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        try:
            print(f"Trying camera with backend {backend}")
            camera = cv2.VideoCapture(0, backend)
            
            # Set explicit camera properties
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Try MJPG format
            
            if camera.isOpened():
                print(f"Camera opened with backend {backend}")
                # Warm up the camera
                for i in range(10):
                    ret, frame = camera.read()
                    if ret:
                        print(f"Camera warmed up successfully on iteration {i}")
                        # Save test image
                        test_path = os.path.join(app.config['UPLOAD_FOLDER'], f'camera_test_{backend}.jpg')
                        cv2.imwrite(test_path, frame)
                        return camera
                    time.sleep(0.1)
                
                print("Camera opened but couldn't read frames")
                camera.release()
            else:
                print(f"Failed to open camera with backend {backend}")
        except Exception as e:
            print(f"Error with backend {backend}: {e}")
    
    print("All camera initialization attempts failed")
    return None

# Ensure we have a clean camera instance
camera = None
if 'camera' in globals() and camera is not None:
    camera.release()

# Initialize fresh camera
camera = initialize_camera()

detection_results = {
    "animal": "",
    "disaster": "",
    "gunshot": "",
    "threat": ""
}

def gen_frames():
    global camera
    
    # Keep track of frame count and last reset time
    frame_count = 0
    last_reset = time.time()
    
    while True:
        # Check if we need to reset the camera (every 60 seconds)
        current_time = time.time()
        if current_time - last_reset > 60:
            print("Periodic camera reset...")
            if camera is not None:
                camera.release()
            camera = initialize_camera()
            last_reset = current_time
            frame_count = 0
            
        # If camera isn't available, provide placeholder
        if camera is None or not camera.isOpened():
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'Camera Not Available', (120, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, 'Attempting to connect...', (120, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(placeholder, timestamp, (10, placeholder.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            _, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.5)  # Don't refresh too quickly
            continue
            
        # Try to read the frame
        success, frame = camera.read()
        frame_count += 1
        
        # If reading fails, recreate camera
        if not success:
            print(f"Frame read failed after {frame_count} frames")
            camera.release()
            camera = initialize_camera()
            continue
        
        # Add overlay information
        cv2.putText(frame, 'Camera Active', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add frame counter for debugging
        cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Control the frame rate to prevent overwhelming the system
        time.sleep(0.05)  # ~20 FPS

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def monitor_system():
    global camera
    while True:
        try:
            if camera is not None and camera.isOpened():
                animal = run_animal_detection(camera)
                disaster = run_disaster_detection(camera)
                gunshot = run_gunshot_detection()

                threat = classify_threat(animal, disaster, gunshot)

                detection_results.update({
                    "animal": animal,
                    "disaster": disaster,
                    "gunshot": gunshot,
                    "threat": threat
                })

                if threat == "Threat Detected":
                    send_telegram_alert(f"⚠️ Threat Detected!\nAnimal: {animal}\nDisaster: {disaster}\nGunshot: {gunshot}")
            else:
                print("Camera not available for monitoring")
                time.sleep(5)  # Wait before trying again
        except Exception as e:
            print(f"Error in monitoring system: {e}")
            time.sleep(5)  # Wait before trying again

@app.route("/")
def index():
    return render_template("index.html", results=detection_results)

@app.route("/status")
def status():
    return jsonify(detection_results)

@app.route("/camera-status")
def camera_status():
    """Endpoint to check camera status and information"""
    global camera
    status_info = {
        "camera_initialized": camera is not None,
        "camera_open": camera is not None and camera.isOpened(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Try to get camera properties if available
    if camera is not None and camera.isOpened():
        status_info.update({
            "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": camera.get(cv2.CAP_PROP_FPS),
            "fourcc": int(camera.get(cv2.CAP_PROP_FOURCC)),
            "fourcc_str": "".join([chr((int(camera.get(cv2.CAP_PROP_FOURCC)) >> 8*i) & 0xFF) for i in range(4)]),
        })
    
    return jsonify(status_info)

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    prediction = predict_animal_from_image(filepath)
    return jsonify({"prediction": prediction, "file_path": filepath})

@app.route("/upload-video", methods=["POST"])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the video file
    run_animal_detection(filepath)
    run_disaster_detection(filepath)
    run_gunshot_detection(filepath)

    return jsonify({"message": "Video processed successfully", "file_path": filepath})

@app.route("/reset-camera")
def reset_camera():
    """Endpoint to manually reset the camera"""
    global camera
    if camera is not None:
        camera.release()
    camera = initialize_camera()
    return jsonify({"message": "Camera reset attempted", "success": camera is not None})

if __name__ == "__main__":
    monitoring_thread = Thread(target=monitor_system, daemon=True)
    monitoring_thread.start()
    app.run(debug=False, threaded=True)  # Use threaded mode for better camera handling