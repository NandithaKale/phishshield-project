from flask import Blueprint, request, jsonify
from app.services.predict_service import predict_url

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400

    url = data['url']

    result = predict_url(url)
    return jsonify(result), 200