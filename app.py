from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from PIL import Image

app = Flask(__name__)

# Enable CORS for a specific origin
CORS(app, origins=["https://video-calling-website-v2.vercel.app"])

# Initialize the emotion detection model
emotion_model = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        # Check if an image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Load the image
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")

        # Run emotion detection
        predictions = emotion_model(image)

        # Return predictions
        return jsonify(predictions), 200

    except Exception as e:
        # Log the error for debugging
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500



