import os
from flask import Flask, send_from_directory, render_template, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Initialize the Flask application
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS for all routes
CORS(app)

# Path to the uploaded TensorFlow Lite model
tflite_model_path = 'deeplabv3.tflite'  # Path to your model file
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Function to segment image using TensorFlow Lite model
def segment_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load image.")
    
    input_image = cv2.resize(image, (257, 257))  # Resize to fit DeepLabV3 model
    input_image = input_image.astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    input_image = np.divide(input_image, 255.0)  # Normalize to 0-1 range

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    output_mask = output.squeeze()  # Remove unnecessary dimensions
    output_mask = np.argmax(output_mask, axis=-1)  # Get the class with the highest score

    return image, output_mask

# Function to apply Gaussian Blur (Bokeh effect) to the background
def apply_bokeh_effect(image, segmentation_mask, blur_strength=15):
    if blur_strength % 2 == 0:
        blur_strength += 1  # Make sure blur_strength is odd

    segmentation_mask_resized = cv2.resize(segmentation_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    subject_mask = segmentation_mask_resized > 0.5  # Binary mask

    subject = np.copy(image)
    subject[~subject_mask] = 0  # Zero out the background

    background = np.copy(image)
    background_blurred = cv2.GaussianBlur(background, (blur_strength, blur_strength), 0)

    final_image = np.where(subject_mask[:, :, None], subject, background_blurred)
    return final_image

# Route for the home page (index.html)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route to handle image upload and saving
@app.route('/api/image/upload', methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file in the static/uploads folder
    upload_folder = os.path.join(app.static_folder, 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)

    # Process the image
    image, segmentation_mask = segment_image(image_path)

    # Generate the processed image and return its URL
    processed_image_path = os.path.join(upload_folder, 'processed_' + file.filename)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_pil.save(processed_image_path)

    return jsonify({
        "success": True,
        "filename": file.filename,
        "original_image": f'/uploads/{file.filename}',
        "processed_image": f'/uploads/processed_{file.filename}'
    })

# Route to apply effect (this handles the blur effect)
@app.route('/api/image/process', methods=["POST"])
def process_image():
    data = request.get_json()

    blur_strength = data.get('blur_strength', 15)
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    # Load the uploaded image
    upload_folder = os.path.join(app.static_folder, 'uploads')
    image_path = os.path.join(upload_folder, filename)

    if not os.path.exists(image_path):
        return jsonify({"error": "File not found"}), 400

    image, segmentation_mask = segment_image(image_path)

    # Apply the blur effect
    final_image = apply_bokeh_effect(image, segmentation_mask, blur_strength)

    # Save the processed image
    processed_image_path = os.path.join(upload_folder, 'processed_' + filename)
    img_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    img_pil.save(processed_image_path)

    return jsonify({
        "success": True,
        "processed_image": f'/uploads/processed_{filename}'
    })

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(app.static_folder, 'uploads')
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
