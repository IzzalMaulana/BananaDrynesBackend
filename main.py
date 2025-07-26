from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import io
import pickle
import os
from flask_cors import CORS
import traceback
import mysql.connector
from datetime import datetime
import pytz

app = Flask(__name__)
# <-- DIUBAH: Mengizinkan akses dari alamat IP publik Anda atau semua alamat (*)
CORS(app, origins=['http://139.59.227.44', '*']) 

# Konstanta minimal confidence untuk validasi gambar pisang
MIN_CONFIDENCE = 76.0 

# Load XGBoost model dari file pickle
try:
    model_path = 'model_xgboost_pisang_result.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} tidak ditemukan!")
        model = None
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Import dan inisialisasi ViT model
try:
    from transformers import ViTImageProcessor, ViTModel
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    vit_model.eval()
    print("ViT model berhasil dimuat!")

    vit_available = True
except Exception as e:
    print(f"Error loading ViT model: {e}")
    vit_available = False

# <-- DIUBAH: Konfigurasi koneksi MySQL dengan user dan password yang benar
db_config = {
    'host': 'localhost',
    'user': 'banana_user',
    'password': 'abcd123', # <-- GANTI DENGAN PASSWORD ANDA
    'database': 'banana_db'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vit_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)
        return features
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        raise e

def get_prediction_confidence(features):
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            max_prob = np.max(probabilities) * 100
            return float(max_prob)
        else:
            return 95.0
    except Exception:
        return 95.0

UPLOAD_FOLDER = 'uploads'
# <-- DIUBAH: Menggunakan cara yang aman untuk membuat folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# <-- DITAMBAHKAN: Rute untuk halaman utama agar tidak 404 Not Found
@app.route('/')
def index():
    return "<h1>Backend BananaDrynes Aktif!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model XGBoost tidak tersedia'}), 500
        if not vit_available:
            return jsonify({'error': 'ViT model tidak tersedia'}), 500
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(save_path)
        img_bytes = open(save_path, 'rb').read()

        features = preprocess_image(img_bytes)
        confidence = get_prediction_confidence(features)

        if confidence < MIN_CONFIDENCE:
            result = {
                'classification': 'Gambar Bukan Pisang',
                'accuracy': round(confidence, 1),
                'drynessLevel': -1,
                'filename': image_file.filename
            }
            # Simpan ke history
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                # <-- DIUBAH: Menghapus is_banana dari query INSERT
                cursor.execute(
                    "INSERT INTO history (filename, classification, accuracy, drynessLevel) VALUES (%s, %s, %s, %s)",
                    (image_file.filename, result['classification'], result['accuracy'], result['drynessLevel'])
                )
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as db_err:
                print(f"Error saving history: {db_err}")
            return jsonify(result)

        pred = model.predict(features)
        pred_class = int(pred[0])
        label_map = {0: "Basah", 1: "Sedang", 2: "Kering"}
        classification = label_map.get(pred_class, "Unknown")

        result = {
            'classification': classification,
            'accuracy': round(confidence, 1),
            'drynessLevel': pred_class,
            'filename': image_file.filename
        }
        # Simpan ke history
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # <-- DIUBAH: Menghapus is_banana dari query INSERT
            cursor.execute(
                "INSERT INTO history (filename, classification, accuracy, drynessLevel) VALUES (%s, %s, %s, %s)",
                (image_file.filename, result['classification'], result['accuracy'], result['drynessLevel'])
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_err:
            print(f"Error saving history: {db_err}")
        return jsonify(result)

    except Exception as e:
        print(f"Error in predict: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vit_available': vit_available
    })

@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM history ORDER BY created_at DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        jakarta = pytz.timezone('Asia/Jakarta')
        for row in rows:
            if isinstance(row['created_at'], datetime):
                row['created_at'] = row['created_at'].astimezone(jakarta).strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(rows)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'error': 'Failed to fetch history'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run()