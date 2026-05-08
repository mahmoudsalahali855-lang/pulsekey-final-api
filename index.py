from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'pulsekey_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'pulsekey_scaler.pkl'))

@app.route('/')
def home():
    return "PulseKey AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
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
        return jsonify({'error': str(e)})

handler = app
