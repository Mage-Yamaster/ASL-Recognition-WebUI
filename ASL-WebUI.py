from flask import Flask, send_from_directory, request, send_file, jsonify
import os
import tempfile
import logging
import ASLR_func as ASLR
from PIL import Image

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

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    img = Image.open(tmp_path)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(tmp_path)

    if no_output:
        os.remove(tmp_path)
        return jsonify(message='Frame received, no output requested')

    # Run classification
    class_desc, class_idx, confidence = ASLR.classify_image(tmp_path, output=overlay)
    output_path = 'outputUI.jpg'

    if not os.path.exists(output_path):
        os.remove(tmp_path)
        return jsonify(error='Output image not found'), 404

    logging.info(f"Classified as {class_desc} (idx={class_idx}, conf={confidence:.2f})")
    os.remove(tmp_path)
    return send_file(output_path, mimetype='image/jpeg', as_attachment=False)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 4040))
    app.run(port=port, debug=True)
