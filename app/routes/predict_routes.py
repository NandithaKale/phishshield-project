from flask import Blueprint, request, jsonify

from app.services.predict_service import predict_url
from shared.url_normalizer import normalize_url

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get("url")

    if not isinstance(url, str) or not url.strip():
        return jsonify({"error": "URL is required"}), 400

    try:
        # Validate/normalize before running any ML or security logic.
        normalize_url(url)
        result = predict_url(url)
        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    except Exception:
        # Avoid exposing internal stack traces to API clients.
        return jsonify({"error": "URL analysis failed"}), 500
