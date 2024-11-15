# backend/app.py
from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)
emotion_model = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image = Image.open(request.files['image'])
    predictions = emotion_model(image)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
