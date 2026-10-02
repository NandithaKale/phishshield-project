from urllib.parse import urlparse

from app.services.ml_model import load_model
from shared.feature_extractor import extract_features
from shared.url_normalizer import normalize_url
from app.services.security_checks import (
    TRUSTED_DOMAINS,
    analyze_url_security,
)
from app.services.xai_service import explain_url


MODEL_THRESHOLD = 0.70
IP_REVIEW_THRESHOLD = 0.55
EXPECTED_FEATURE_COUNT = 19


def get_base_domain(url):
    normalized = normalize_url(url)
    hostname = (urlparse(normalized.lower()).hostname or "").strip(".")

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def _final_decision(model_probability, security):
    """
    Hybrid decision layer.

    - SHAP explains the ML model.
    - Deterministic security rules handle high-confidence URL anomalies.
    - A look-alike domain is a strong phishing signal.
    - An IP address raises the decision threshold sensitivity but is not
      treated as automatically malicious.
    """
    if security["lookalike_domain"]["is_lookalike"]:
        return {
            "is_phishing": True,
            "final_probability": max(float(model_probability), 0.90),
            "risk_level": "high",
            "decision_reason": "Look-alike domain detected.",
        }

    if model_probability >= MODEL_THRESHOLD:
        return {
            "is_phishing": True,
            "final_probability": float(model_probability),
            "risk_level": "high",
            "decision_reason": "ML phishing probability exceeded the threshold.",
        }

    if security["is_ip_address"] and model_probability >= IP_REVIEW_THRESHOLD:
        return {
            "is_phishing": True,
            "final_probability": max(float(model_probability), 0.70),
            "risk_level": "high",
            "decision_reason": "Direct IP address combined with elevated ML risk.",
        }

    if security["is_ip_address"]:
        return {
            "is_phishing": False,
            "final_probability": float(model_probability),
            "risk_level": "warning",
            "decision_reason": "Direct IP address detected; manual caution is advised.",
        }

    return {
        "is_phishing": False,
        "final_probability": float(model_probability),
        "risk_level": "low",
        "decision_reason": "No security override and ML probability is below threshold.",
    }


def predict_url(url):
    normalized_url = normalize_url(url)
    model = load_model()

    # The existing .pkl was trained with the old 17-feature vector. The
    # enhanced model must be retrained with the new 19-feature extractor.
    if getattr(model, "n_features_in_", EXPECTED_FEATURE_COUNT) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "The saved URL model uses the old feature set. "
            "Run ml_model/src/train_url_model.py to retrain it."
        )

    features = extract_features(normalized_url)
    base_domain = get_base_domain(normalized_url)
    security = analyze_url_security(normalized_url)

    probability = float(model.predict_proba([features])[0][1])

    # Exact trusted domains retain the existing trusted-domain behavior,
    # but deterministic security signals are checked first.
    if base_domain in TRUSTED_DOMAINS and not security["risk_flags"]:
        explanation = [{
            "feature": "Trusted domain",
            "value": base_domain,
            "impact": -1.0,
            "reason": "This exact domain is in the trusted-domain list.",
            "source": "trusted"
        }]

        return {
            "url": url,
            "normalized_url": normalized_url,
            "is_phishing": False,
            "confidence": 0.99,
            "risk_level": "low",
            "decision_reason": "Exact trusted domain match.",
            "security_analysis": security,
            "explanation": explanation,
        }

    decision = _final_decision(probability, security)
    explanation = explain_url(normalized_url)

    # Add deterministic security reasons alongside SHAP explanations.
    if security["is_ip_address"]:
        explanation.insert(0, {
            "feature": "IP-address detection",
            "value": True,
            "impact": 0.0,
            "reason": "The hostname is a direct IP address rather than a domain name.",
            "source": "security"
        })

    if security["lookalike_domain"]["is_lookalike"]:
        match = security["lookalike_domain"]["matched_domain"]
        similarity = security["lookalike_domain"]["similarity"]
        explanation.insert(0, {
            "feature": "Look-alike domain detection",
            "value": match,
            "impact": 0.0,
            "reason": (
                f"The domain closely resembles {match} "
                f"(similarity={similarity:.2f})."
            ),
            "source": "security",
        })

    return {
        "url": url,
        "normalized_url": normalized_url,
        "is_phishing": decision["is_phishing"],
        "confidence": (decision["final_probability"] if decision["is_phishing"] else 1.0 - decision["final_probability"]),
        "ml_probability": probability,
        "risk_level": decision["risk_level"],
        "decision_reason": decision["decision_reason"],
        "security_analysis": security,
        "explanation": explanation,
    }
