from flask import Flask, render_template, Response, request, send_from_directory
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import easyocr
from collections import Counter
import pandas as pd
from rapidfuzz import fuzz, process  # Import RapidFuzz for fuzzy matching

# Initialize the Flask app
app = Flask(__name__)

# Directory to save processed results
RESULT_FOLDER = 'results'
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Path to the dataset CSV
CSV_FILE_PATH = "E:/Flipkart/Product.csv"  # Ensure this is correct

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Global variable for webcam feed
camera = cv2.VideoCapture(0)  # Default camera, change index if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    # Capture a frame from the webcam
    success, frame = camera.read()
    if not success:
        return "Failed to capture image", 500

    # Save the frame temporarily
    temp_image_path = os.path.join(RESULT_FOLDER, 'captured_image.jpg')
    cv2.imwrite(temp_image_path, frame)

    # Process the image for OCR and item counting
    ocr_result, processed_img_path, match_counts = process_image(temp_image_path)

    return render_template(
        'result.html',
        result=ocr_result,
        image_url=processed_img_path,
        match_counts=match_counts
    )

def normalize_text(text):
    """ Normalize the text for OCR and CSV matching """
    return "".join(char for char in text.upper() if char.isalnum() or char.isspace()).strip()

def process_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Resize and enhance the image
    img = img.resize((800, 600), Image.LANCZOS)
    gray_img = img.convert('L')
    contrast_img = ImageEnhance.Contrast(gray_img).enhance(2.0)
    denoised_img = contrast_img.filter(ImageFilter.MedianFilter(size=3))

    # Convert to numpy array for EasyOCR
    denoised_img_np = np.array(denoised_img)

    # Try OCR on multiple rotations
    rotations = [0, 90, 180, 270]
    best_text = ""
    for angle in rotations:
        img_rotated = img.rotate(angle, expand=True)
        denoised_img_np_rotated = np.array(img_rotated)
        try:
            result = reader.readtext(denoised_img_np_rotated)
            text_from_image = " ".join([res[1] for res in result])
            if len(text_from_image) > len(best_text):
                best_text = text_from_image
        except Exception as e:
            print(f"Error in OCR at {angle}°: {e}")

    print(f"Raw OCR Text: {best_text}")  # Debug log

    # Normalize OCR result
    normalized_text = normalize_text(best_text)
    ocr_words = normalized_text.split()
    print(f"Normalized OCR Words: {ocr_words}")  # Debug log

    # Load CSV data (ensure it contains a 'Product' column)
    try:
        csv_data = pd.read_csv(CSV_FILE_PATH)
        csv_products = [normalize_text(product) for product in csv_data['Product'].tolist()]
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return "Failed to load CSV", None, {}

    print(f"CSV Products: {csv_products}")  # Debug log

    # Matching products from OCR with CSV products
    match_counts = {}
    match_details = []  # Store details of matches

    for word in ocr_words:
        best_match, score, _ = process.extractOne(word, csv_products, scorer=fuzz.ratio)
        print(f"Word: {word}, Best Match: {best_match}, Score: {score}")  # Debug log
        if score >= 85:  # High confidence
            match_counts[best_match] = match_counts.get(best_match, 0) + 1
            match_details.append((word, best_match, score, "High Confidence"))
        elif score >= 70:  # Moderate confidence
            match_counts[best_match] = match_counts.get(best_match, 0) + 1
            match_details.append((word, best_match, score, "Moderate Confidence"))
        else:
            match_details.append((word, "No Match", score, "Low Confidence"))

    print(f"Matched Counts: {match_counts}")  # Debug log
    print(f"Match Details: {match_details}")  # Debug log

    # Save processed image
    processed_img_path = os.path.join(RESULT_FOLDER, 'processed_image.jpg')
    denoised_img.save(processed_img_path)

    return best_text if best_text else "No text found", processed_img_path, match_counts

@app.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
