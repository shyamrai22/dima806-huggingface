from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
emotion_model = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Load and preprocess the image
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")

        # Run the model
        predictions = emotion_model(image)

        # Return the top predictions
        return jsonify(predictions), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
