from flask import Blueprint, jsonify
from app.models.prediction_model import Prediction

history_bp = Blueprint('history', __name__)

@history_bp.route('/history', methods=['GET'])
def get_history():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()

    result = []

    for r in records:
        result.append({
            "url": r.url,
            "is_phishing": r.is_phishing,
            "confidence": r.confidence,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })

    return jsonify(result)