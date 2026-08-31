import shap
import numpy as np

from app.services.ml_model import load_model
from shared.feature_extractor import extract_features


FEATURE_NAMES = [
    "URL length",
    "Number of dots",
    "Number of hyphens",
    "Number of slashes",
    "Number of equal signs",
    "HTTPS present",
    "HTTP without HTTPS",
    "@ symbol present",
    "Number of digits",
    "Suspicious word present",
    "Number of suspicious words",
    "Suspicious TLD",
    "Multiple subdomains",
    "Hyphen in domain",
    "Digit in domain",
    "Domain length",
    "URL length > 60"
]


FEATURE_REASONS = {
    "URL length": "The URL is unusually long.",
    "Number of dots": "The URL contains multiple dots.",
    "Number of hyphens": "The URL contains multiple hyphens.",
    "Number of slashes": "The URL contains multiple path separators.",
    "Number of equal signs": "The URL contains query parameters.",
    "HTTPS present": "The URL uses HTTPS.",
    "HTTP without HTTPS": "The URL uses HTTP instead of HTTPS.",
    "@ symbol present": "The URL contains an @ symbol.",
    "Number of digits": "The URL contains numeric characters.",
    "Suspicious word present": "Suspicious keywords were detected.",
    "Number of suspicious words": "Multiple suspicious keywords were detected.",
    "Suspicious TLD": "The domain uses a potentially suspicious TLD.",
    "Multiple subdomains": "The domain contains multiple subdomains.",
    "Hyphen in domain": "The domain contains a hyphen.",
    "Digit in domain": "The domain contains numeric characters.",
    "Domain length": "The domain is relatively long.",
    "URL length > 60": "The URL exceeds 60 characters."
}


def explain_url(url, top_n=5):
    model = load_model()
    features = extract_features(url)

    explainer = shap.TreeExplainer(model)
    X = np.array([features], dtype=float)
    shap_values = explainer.shap_values(X)

    # SHAP 0.52+ may return a 3D array for classifiers:
    # (samples, features, outputs)
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

        if getattr(values, "ndim", 1) == 2:
            values = values[:, 1]

    feature_impacts = []

    for name, value, shap_value in zip(
        FEATURE_NAMES,
        features,
        values
    ):
        feature_impacts.append({
            "feature": name,
            "value": value,
            "shap_value": float(shap_value),
            "impact": abs(float(shap_value))
        })

    feature_impacts.sort(
        key=lambda x: x["impact"],
        reverse=True
    )

    explanations = []

    for item in feature_impacts[:top_n]:
        explanations.append({
            "feature": item["feature"],
            "value": item["value"],
            "impact": round(item["shap_value"], 4),
            "reason": FEATURE_REASONS[item["feature"]]
        })

    return explanations