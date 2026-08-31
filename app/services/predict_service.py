from app.services.ml_model import load_model
from shared.feature_extractor import extract_features
from app.services.xai_service import explain_url


def predict_url(url):
    model = load_model()

    features = extract_features(url)

    probability = model.predict_proba([features])[0][1]

    # Custom threshold
    threshold = 0.7
    is_phishing = probability >= threshold

    # Generate XAI explanation
    explanation = explain_url(url)

    return {
        "url": url,
        "is_phishing": bool(is_phishing),
        "confidence": float(probability),
        "explanation": explanation
    }