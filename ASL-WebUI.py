#!/usr/bin/python3
from flask import Flask, send_from_directory, request, send_file, jsonify
import os
import sys
import tempfile
import logging
from PIL import Image
import jetson_inference
import jetson_utils
import time

net = None

def truncate_float(number, decimals):
    """Truncates a float to a specified number of decimal places."""
    factor = 10 ** decimals
    return int(number * factor) / factor

def classify_and_overlay(image_path, overlay=False):
    """
    Classifies an image using the pre-loaded model and optionally overlays the result.
    """
    if net is None:
        return "Error: Model not loaded", -1, 0.0

    try:
        img = jetson_utils.loadImage(image_path)
        class_idx, confidence = net.Classify(img)
        class_desc = net.GetClassDesc(class_idx)

        if overlay:
            text = f"{truncate_float(confidence * 100, 2)}% {class_desc}"
            # Calculate font size, ensure it's not too large
            Tsize = min(30, img.width * 2 / len(text) - 2)
            font = jetson_utils.cudaFont(size=Tsize)
            font.OverlayText(img, img.width, img.height, text, 5, 5,
                             color=(0, 240, 255, 180), background=(0, 0, 0, 120))

            # Save the image with overlay
            jetson_utils.saveImage("outputUI.jpg", img)

        return class_desc, class_idx, confidence
    except Exception as e:
        logging.error(f"Classification failed: {e}")
        return "Error", -1, 0.0

# --- Flask App ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return send_from_directory('.', 'WebUI.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'frame' not in request.files:
        return jsonify(error='No frame uploaded'), 400

    file = request.files['frame']
    overlay = request.form.get('overlay') == '1'
    no_output = request.form.get('noOutput') == '1'

    logging.info(f"Overlay: {overlay}, No Output: {no_output}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        img = Image.open(tmp_path)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(tmp_path)
        img.close()

        if no_output:
            return jsonify(message='Frame received, no output requested')

        class_desc, class_idx, confidence = classify_and_overlay(tmp_path, overlay=overlay)

        if class_idx == -1:
            return jsonify(error=class_desc), 500

        output_path = 'outputUI.jpg'
        if not os.path.exists(output_path) and overlay:
            return jsonify(error='Output image not found'), 404

        logging.info(f"Classified as {class_desc} (idx={class_idx}, conf={confidence:.2f})")
        
        return send_file(output_path, mimetype='image/jpeg', as_attachment=False)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists('outputUI.jpg'):
            os.remove('outputUI.jpg')

if __name__ == '__main__':
    print("Your App link will appear in your console.")
    time.sleep(0.5)
    port = int(os.getenv('PORT', 4040))

    try:
        pathName = os.path.dirname(__file__) + "/googlenet-ASL.onnx"
        net = jetson_inference.imageNet("googlenet",
                                        model=pathName,
                                        input_blob="input_0",
                                        output_blob="output_0",
                                        labels="labels.txt")
        
    except Exception as e:
        logging.error(f"Failed to load the model: {e}")
        net = None
        sys.exit(1)

    app.run(port=port, debug=True)
