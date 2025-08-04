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
CORS(app, origins=['http://bananadrynes.my.id', 'http://www.bananadrynes.my.id', '*'], methods=['GET', 'POST', 'DELETE', 'OPTIONS']) 

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
    'password': 'abc123', # <-- GANTI DENGAN PASSWORD ANDA YANG BENAR
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

@app.route('/test-delete/<int:history_id>', methods=['DELETE'])
def test_delete(history_id):
    """Test endpoint untuk debugging delete operation"""
    print(f"Test DELETE request received for history ID: {history_id}")
    return jsonify({
        'message': 'Test DELETE endpoint working',
        'history_id': history_id,
        'method': 'DELETE'
    }), 200

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
        
        # Logika prediksi dipisahkan agar lebih rapi
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            confidence = np.max(probabilities) * 100
        else:
            confidence = 95.0

        if confidence < MIN_CONFIDENCE:
            classification = 'Gambar Bukan Pisang'
            dryness_level = -1
        else:
            pred = model.predict(features)
            pred_class = int(pred[0])
            label_map = {0: "Basah", 1: "Sedang", 2: "Kering"}
            classification = label_map.get(pred_class, "Unknown")
            dryness_level = pred_class

        result = {
            'classification': classification,
            'accuracy': round(float(confidence), 1),
            'drynessLevel': dryness_level,
            'filename': image_file.filename
        }

        # Simpan ke history (logika penyimpanan dijadikan satu)
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
            # Opsional: Anda bisa mengembalikan error jika penyimpanan gagal
            # return jsonify({'error': 'Gagal menyimpan hasil ke database'}), 500

        return jsonify(result)

    except Exception as e:
        print(f"!!! TERJADI ERROR DI FUNGSI PREDICT: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

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

@app.route('/history/<int:history_id>', methods=['DELETE'])
def delete_history(history_id):
    print(f"DELETE request received for history ID: {history_id}")
    try:
        conn = get_db_connection()
        if not conn:
            print("Database connection failed")
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Get filename before deleting for cleanup
        cursor.execute("SELECT filename FROM history WHERE id = %s", (history_id,))
        result = cursor.fetchone()
        
        if not result:
            print(f"History record with ID {history_id} not found")
            cursor.close()
            conn.close()
            return jsonify({'error': 'History record not found'}), 404
        
        filename = result[0]
        print(f"Found filename: {filename}")
        
        # Delete from database
        cursor.execute("DELETE FROM history WHERE id = %s", (history_id,))
        
        if cursor.rowcount == 0:
            print(f"No rows affected when deleting ID {history_id}")
            cursor.close()
            conn.close()
            return jsonify({'error': 'Failed to delete history record'}), 500
        
        print(f"Successfully deleted {cursor.rowcount} record(s) from database")
        conn.commit()
        cursor.close()
        conn.close()
        
        # Try to delete the uploaded file
        try:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {filename}")
            else:
                print(f"File not found: {file_path}")
        except Exception as file_err:
            print(f"Warning: Could not delete file {filename}: {file_err}")
        
        print(f"DELETE operation completed successfully for ID: {history_id}")
        return jsonify({'message': 'History deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting history: {e}")
        print(traceback.format_exc())
        return jsonify({'error': 'Failed to delete history'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)