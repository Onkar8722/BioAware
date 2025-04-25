from flask import Flask, render_template, jsonify, request, Response
from threading import Thread
import os
import cv2
from werkzeug.utils import secure_filename

from services.animal_detection import predict_animal_from_image, run_animal_detection, model as animal_model, categories as animal_categories
from services.disaster_detection import run_disaster_detection
from services.gunshot_detection import run_gunshot_detection
from utils.threat_classifier import classify_threat
from alerts.notifier import send_telegram_alert

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

camera = cv2.VideoCapture(0)

detection_results = {
    "animal": "",
    "disaster": "",
    "gunshot": "",
    "threat": ""
}

def monitor_system():
    while True:
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

@app.route("/")
def index():
    return render_template("index.html", results=detection_results)

@app.route("/status")
def status():
    return jsonify(detection_results)

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

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    monitoring_thread = Thread(target=monitor_system, daemon=True)
    monitoring_thread.start()
    app.run(debug=True)
