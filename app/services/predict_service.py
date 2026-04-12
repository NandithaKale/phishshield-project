from app.services.ml_model import load_model
from shared.feature_extractor import extract_features

def predict_url(url):
    model = load_model()

    features = extract_features(url)

    probability = model.predict_proba([features])[0][1]

    # 🔥 FIX: Custom threshold (not default 0.5)
    threshold = 0.7
    is_phishing = probability >= threshold

    return {
        "url": url,
        "is_phishing": bool(is_phishing),
        "confidence": float(probability)
    }