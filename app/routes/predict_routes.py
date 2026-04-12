from flask import Blueprint, request, jsonify
from app.services.predict_service import predict_url

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    result = predict_url(url)
    return jsonify(result)