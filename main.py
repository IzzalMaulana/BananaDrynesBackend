# main.py

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
import random

app = Flask(__name__)
CORS(app, origins=['http://bananadrynes.my.id', 'http://www.bananadrynes.my.id']) 

# Konstanta
MIN_CONFIDENCE = 76.0 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model saat aplikasi dimulai
model = None
vit_model = None
extractor = None
vit_available = False

try:
    model_path = 'model_xgboost_pisang_result.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model XGBoost berhasil dimuat!")
    else:
        print(f"Error: Model file {model_path} tidak ditemukan!")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")

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

# Konfigurasi koneksi MySQL
db_config = {
    'host': 'localhost',
    'user': 'banana_user',
    'password': 'abc123', # <-- PASTIKAN INI BENAR
    'database': 'banana_db'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

@app.route('/')
def index():
    return "<h1>Backend BananaDrynes Aktif!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not vit_available:
        return jsonify({'error': 'Satu atau lebih model tidak tersedia'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(save_path)
        img_bytes = open(save_path, 'rb').read()
        features = preprocess_image(img_bytes)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            confidence = np.max(probabilities) * 100
        else:
            confidence = 95.0
        
        recommendations = {
            "Basah": [
                "Kadar air masih tinggi. Lanjutkan proses pengeringan.",
                "Sempurna untuk diolah menjadi adonan kue pisang. Untuk membuatnya kering, butuh waktu penjemuran lebih lama."
            ],
            "Sedang": [
                "Hampir kering. Lanjutkan pengeringan untuk hasil yang lebih renyah.",
                "Sudah setengah jalan! Cocok untuk pisang sale yang masih kenyal. Jemur sedikit lebih lama jika Anda ingin lebih garing.",
                "Tekstur saat ini ideal untuk dijadikan isian roti atau topping. Untuk daya simpan maksimal, lanjutkan pengeringan."
            ],
            "Kering": [
                "Sempurna! Tingkat kekeringan ideal telah tercapai. Segera simpan dalam wadah kedap udara untuk menjaga kerenyahannya.",
                "Hasil terbaik! Pisang Anda siap dinikmati atau dijual. Pastikan disimpan di tempat sejuk dan kering."
            ]
        }
        
        recommendation_text = ""
        if confidence < MIN_CONFIDENCE:
            classification = 'Gambar Bukan Pisang'
            dryness_level = -1
            recommendation_text = "Gambar yang diunggah sepertinya bukan pisang. Coba gunakan gambar lain."
        else:
            pred = model.predict(features)
            pred_class = int(pred[0])
            label_map = {0: "Basah", 1: "Sedang", 2: "Kering"}
            classification = label_map.get(pred_class, "Unknown")
            dryness_level = pred_class
            if classification in recommendations:
                recommendation_text = random.choice(recommendations[classification])

        result = {
            'classification': classification,
            'accuracy': round(float(confidence), 1),
            'drynessLevel': dryness_level,
            'filename': image_file.filename,
            'recommendation': recommendation_text
        }

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO history (filename, classification, accuracy, drynessLevel) VALUES (%s, %s, %s, %s)",
                (result['filename'], result['classification'], result['accuracy'], result['drynessLevel'])
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_err:
            print(f"!!! GAGAL MENYIMPAN KE DATABASE: {db_err}")
            
        return jsonify(result)

    except Exception as e:
        print(f"!!! TERJADI ERROR DI FUNGSI PREDICT: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    # ... (fungsi ini tidak perlu diubah) ...
    pass

@app.route('/history/<int:id>', methods=['DELETE'])
def delete_history(id):
    # ... (fungsi ini tidak perlu diubah) ...
    pass

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)