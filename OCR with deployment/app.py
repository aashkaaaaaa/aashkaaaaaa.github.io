from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import easyocr
from collections import Counter
import pandas as pd
from rapidfuzz import fuzz, process
import base64
import io

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

@app.route('/')
def index():
    return render_template('index.html')

def normalize_text(text):
    """ Normalize the text for OCR and CSV matching """
    return "".join(char for char in text.upper() if char.isalnum() or char.isspace()).strip()

def process_image(image):
    # Resize and enhance the image
    img = image.resize((800, 600), Image.LANCZOS)
    gray_img = img.convert('L')
    contrast_img = ImageEnhance.Contrast(gray_img).enhance(2.0)
    denoised_img = contrast_img.filter(ImageFilter.MedianFilter(size=3))

    # Try OCR on multiple rotations
    rotations = [0, 90, 180, 270]
    best_text = ""
    for angle in rotations:
        img_rotated = img.rotate(angle, expand=True)
        try:
            result = reader.readtext(np.array(img_rotated))
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

    # Load CSV data
    try:
        csv_data = pd.read_csv(CSV_FILE_PATH)
        csv_products = [normalize_text(product) for product in csv_data['Product'].tolist()]
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return "Failed to load CSV", {}

    # Matching products from OCR with CSV products
    match_counts = {}
    match_details = []

    for word in ocr_words:
        best_match, score, _ = process.extractOne(word, csv_products, scorer=fuzz.ratio)
        if score >= 85:
            match_counts[best_match] = match_counts.get(best_match, 0) + 1
            match_details.append((word, best_match, score, "High Confidence"))
        elif score >= 70:
            match_counts[best_match] = match_counts.get(best_match, 0) + 1
            match_details.append((word, best_match, score, "Moderate Confidence"))
        else:
            match_details.append((word, "No Match", score, "Low Confidence"))

    print(f"Matched Counts: {match_counts}")
    print(f"Match Details: {match_details}")

    return best_text if best_text else "No text found", match_counts

@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    try:
        data = request.json
        image_data = data['image']

        # Decode the base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))

        # Process the image
        ocr_result, match_counts = process_image(image)

        return jsonify({
            'result': ocr_result,
            'match_counts': match_counts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
