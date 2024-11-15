from flask import Flask, render_template, Response, request, jsonify
import cv2
from transformers import pipeline
from PIL import Image
import logging
from collections import defaultdict

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load Hugging Face emotion detection model (image-classification pipeline)
emotion_model = pipeline('image-classification', model='dima806/facial_emotions_image_detection')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Store camera instances by user id
cameras = defaultdict(lambda: None)  # Default to None if not found

def generate_frames(camera_id):
    camera = cameras[camera_id]
    while camera and camera.isOpened():
        success, frame = camera.read()
        if not success:
            logging.error(f"Failed to read frame from camera {camera_id}")
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = frame[y:y + h, x:x + w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)

                try:
                    result = emotion_model(pil_image)
                    label = result[0]['label']
                    confidence = result[0]['score']
                except Exception as e:
                    logging.error(f"Error during emotion detection for camera {camera_id}: {e}")
                    label = "Unknown"
                    confidence = 0.0

                cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.error(f"Failed to encode frame for camera {camera_id} as JPEG")
                break
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam/<camera_id>', methods=['POST'])
def start_webcam(camera_id):
    if cameras[camera_id] is not None and cameras[camera_id].isOpened():
        return jsonify({"message": "Webcam is already running"}), 200

    try:
        camera = cv2.VideoCapture(0)  # Open the first available webcam
        if not camera.isOpened():
            logging.error(f"Failed to open webcam for camera {camera_id}")
            return jsonify({"error": "Failed to open webcam"}), 500
        cameras[camera_id] = camera
        return jsonify({"message": "Webcam started"}), 200
    except Exception as e:
        logging.error(f"Error starting webcam for camera {camera_id}: {e}")
        return jsonify({"error": "Failed to start webcam"}), 500

@app.route('/stop_webcam/<camera_id>', methods=['POST'])
def stop_webcam(camera_id):
    try:
        if cameras[camera_id] and cameras[camera_id].isOpened():
            cameras[camera_id].release()
            cameras[camera_id] = None
            return jsonify({"message": "Webcam stopped"}), 200
        else:
            return jsonify({"message": "Webcam is not running"}), 200
    except Exception as e:
        logging.error(f"Error stopping webcam for camera {camera_id}: {e}")
        return jsonify({"error": "Failed to stop webcam"}), 500

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")

        predictions = emotion_model(image)

        return jsonify(predictions), 200
    except Exception as e:
        logging.error(f"Error during emotion detection: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
