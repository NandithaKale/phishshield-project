from urllib.parse import urlparse

from app.services.ml_model import load_model
from shared.feature_extractor import extract_features
from app.services.xai_service import explain_url


TRUSTED_DOMAINS = {
    "google.com",
    "microsoft.com",
    "github.com",
    "wikipedia.org",
    "amazon.in",
}


def get_base_domain(url):
    hostname = (urlparse(url.lower()).hostname or "").strip(".")

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def predict_url(url):
    model = load_model()
    features = extract_features(url)
    base_domain = get_base_domain(url)

    # Known legitimate domains
    if base_domain in TRUSTED_DOMAINS:
        explanation = [{
            "feature": "Trusted domain",
            "value": base_domain,
            "impact": -1.0,
            "reason": "This exact domain is in the trusted-domain list."
        }]

        return {
            "url": url,
            "is_phishing": False,
            "confidence": 0.99,
            "explanation": explanation
        }

    probability = model.predict_proba([features])[0][1]

    threshold = 0.7
    is_phishing = probability >= threshold

    explanation = explain_url(url)

    return {
        "url": url,
        "is_phishing": bool(is_phishing),
        "confidence": float(probability),
        "explanation": explanation
    }
