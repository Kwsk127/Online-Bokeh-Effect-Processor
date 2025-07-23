# Bokeh Effect Image Processor

This project is a web-based image processor that applies a Gaussian blur (Bokeh effect) to the background of uploaded images using TensorFlow Lite's DeepLabV3 model for image segmentation.

## Features

- Upload an image for background segmentation using DeepLabV3.
- Apply Gaussian blur (Bokeh effect) to the background.
- Adjust blur strength using a slider.
- Download processed images.
- Simple user interface with drag-and-drop functionality for easy image upload.

## Technologies Used

- **Flask**: Web framework for building the backend.
- **TensorFlow Lite**: Lightweight version of TensorFlow for deploying models on mobile and embedded devices.
- **OpenCV**: Image processing library used for resizing, masking, and applying the blur effect.
- **PIL**: Python Imaging Library to handle image saving and format conversions.

## Setup

### Prerequisites

1. Python 3.x
2. TensorFlow Lite model (`deeplabv3.tflite`)
3. Libraries:
   - Flask
   - Flask-CORS
   - TensorFlow
   - OpenCV
   - PIL (Pillow)
   - Numpy

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Kwsk127/Online-Bokeh-Effect-Processor

2. Install required dependencies:
   ```bash
   pip install flask flask-cors tensorflow opencv-python numpy Pillow

3. Place the TensorFlow Lite model deeplabv3.tflite in the project directory.
   [DeepLabV3 from Kaggle](https://www.kaggle.com/models/tensorflow/deeplabv3/).

4. Run the Flask app:
   ```bash
   python app.py

###Usage

1. Open the app in your browser.

2. Upload an image by clicking the upload area or dragging and dropping a file.

3. Once the image is uploaded, it will be processed and displayed alongside the original image.

4. Use the blur strength slider to adjust the background blur intensity.

5. Click on the "Apply Effect" button to process the image.

6. Download the processed image by clicking the "Download" button.
   
### File Structure


```bash
   .
   ├── app.py                # Flask backend code 
   ├── static/
   │   ├── uploads/          # Folder to store uploaded images
   ├── templates/
   │   └── index.html        # Frontend HTML template
   └── requirements.txt      # Python dependencies
   └── deeplabv3.tflite      # TensorFlow Lite model

   
