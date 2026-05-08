from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# تحميل الموديل - تأكد أن الملف في المجلد الرئيسي
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return "PulseKey AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        # تحويل البيانات لمصفوفة NumPy
        input_data = np.array(data).reshape(1, -1)
        
        # التنبؤ
        prediction = model.predict(input_data)
        
        return jsonify({
            'status': 'success',
            'risk_level_prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)})
