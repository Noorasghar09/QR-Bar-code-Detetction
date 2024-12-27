
from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from tensorflow.keras.models import load_model
import joblib
from skimage.feature import hog
import sqlite3

app = Flask(__name__)

# Load models
qr_barcode_model = load_model('models/barcode_qr_model.h5')
rf_classifier = joblib.load('models/rf_qr_code_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Database setup
def init_db():
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data TEXT UNIQUE,
                        status TEXT,
                        classification TEXT
                      )''')
    conn.commit()
    conn.close()

def save_report(data, status, classification):
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE data = ?", (data,))
    existing_report = cursor.fetchone()
    if existing_report is None:  # Only save if the QR code is new
        cursor.execute("INSERT INTO reports (data, status, classification) VALUES (?, ?, ?)",
                       (data, status, classification))
        conn.commit()
    conn.close()

def delete_report(report_id):
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()

@app.route('/view_reports')
def view_reports():
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports")
    reports = cursor.fetchall()
    conn.close()
    return render_template('view_reports.html', reports=reports)

@app.route('/delete_report/<int:report_id>', methods=['POST'])
def delete_report_route(report_id):
    delete_report(report_id)
    return redirect(url_for('view_reports'))

# Preprocessing and QR logic
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def check_maliciousness(qr_region):
    qr_region_resized = cv2.resize(qr_region, (32, 32))
    fd, _ = hog(qr_region_resized, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True)
    prediction = rf_classifier.predict([fd])
    label = label_encoder.inverse_transform(prediction)[0]
    return label

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/real_time_scan')
def real_time_scan():
    return render_template('real_time_scan.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

# @app.route('/scan_image', methods=['POST'])
# def scan_image():
#     image = request.files['qr_image']
#     img_bytes = image.read()
#     img_array = np.frombuffer(img_bytes, np.uint8)
#     frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#     if frame is None:
#         return jsonify({'error': 'Unable to process the image.'})

#     preprocessed_frame = preprocess_frame(frame)
#     decoded_objects = decode(preprocessed_frame)

#     data = ''
#     result = 'No QR detected'
#     status = 'Safe'

#     for obj in decoded_objects:
#         points = obj.polygon
#         if points:
#             points = [(int(p.x), int(p.y)) for p in points]
#             data = obj.data.decode('utf-8')
#             x_min = max(0, min(p[0] for p in points))
#             y_min = max(0, min(p[1] for p in points))
#             x_max = min(frame.shape[1], max(p[0] for p in points))
#             y_max = min(frame.shape[0], max(p[1] for p in points))

#             qr_region = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
#             label = check_maliciousness(qr_region)
#             result = 'Malicious' if label == 'malicious' else 'Safe'
#             status = result

#             if status == 'Malicious':
#                 save_report(data, status, result)

#     return jsonify({
#         'data': data,
#         'classification': result,
#         'status': status
#     })



@app.route('/scan_image', methods=['POST'])
def scan_image():
    image = request.files['qr_image']
    img_bytes = image.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Unable to process the image.'})

    preprocessed_frame = preprocess_frame(frame)
    decoded_objects = decode(preprocessed_frame)

    data = ''
    result = 'No QR detected'
    status = 'Safe'

    for obj in decoded_objects:
        points = obj.polygon
        if points:
            points = [(int(p.x), int(p.y)) for p in points]
            data = obj.data.decode('utf-8')
            x_min = max(0, min(p[0] for p in points))
            y_min = max(0, min(p[1] for p in points))
            x_max = min(frame.shape[1], max(p[0] for p in points))
            y_max = min(frame.shape[0], max(p[1] for p in points))

            qr_region = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
            label = check_maliciousness(qr_region)
            result = 'Malicious' if label == 'malicious' else 'Safe'
            status = result

            if status == 'Malicious':
                save_report(data, status, result)
                return jsonify({
                    'alert': 'This image contains malicious content!',
                    'status': status
                })

    return jsonify({
        'data': data,
        'classification': result,
        'status': status
    })



if __name__ == "__main__":
    init_db()
    app.run(debug=True)
