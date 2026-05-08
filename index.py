from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import glob

# التصحيح هنا: لازم __name__
app = Flask(__name__)

def load_resources():
    try:
        model_files = glob.glob("**/pulsekey_model.pkl", recursive=True) + glob.glob("pulsekey_model.pkl")
        scaler_files = glob.glob("**/pulsekey_scaler.pkl", recursive=True) + glob.glob("pulsekey_scaler.pkl")
        
        if model_files and scaler_files:
            m = joblib.load(model_files[0])
            s = joblib.load(scaler_files[0])
            return m, s
        return None, None
    except Exception as e:
        return None, None

model, scaler = load_resources()

@app.route('/')
def home():
    return "PulseKey AI API is Online and Ready!"

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        model, scaler = load_resources()
        if model is None:
            return jsonify({"error": "Model files not found"}), 500
    try:
        data = request.json['data']
        input_data = np.array(data).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        return jsonify({
            'status': 'success',
            'risk_level': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

handler = app
