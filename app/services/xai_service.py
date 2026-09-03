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


def get_feature_reason(feature, value):
    """
    Generate an explanation based on the actual feature value.
    This avoids misleading statements such as saying that
    a URL contains digits when the digit count is actually zero.
    """

    if feature == "URL length":
        if value > 60:
            return f"The URL is long ({value} characters)."
        elif value > 30:
            return f"The URL has a moderate length ({value} characters)."
        else:
            return f"The URL is relatively short ({value} characters)."

    if feature == "Number of dots":
        if value == 0:
            return "No dots were detected in the URL."
        return f"The URL contains {value} dot(s)."

    if feature == "Number of hyphens":
        if value == 0:
            return "No hyphens were detected in the URL."
        return f"The URL contains {value} hyphen(s)."

    if feature == "Number of slashes":
        return f"The URL contains {value} slash(es)."

    if feature == "Number of equal signs":
        if value == 0:
            return "No query-parameter separators were detected."
        return f"The URL contains {value} query-parameter separator(s)."

    if feature == "HTTPS present":
        if value == 1:
            return "The URL uses HTTPS."
        return "The URL does not use HTTPS."

    if feature == "HTTP without HTTPS":
        if value == 1:
            return "The URL uses HTTP instead of HTTPS."
        return "The URL is not using plain HTTP."

    if feature == "@ symbol present":
        if value == 1:
            return "The URL contains an @ symbol."
        return "No @ symbol was detected."

    if feature == "Number of digits":
        if value == 0:
            return "No numeric characters were detected."
        return f"The URL contains {value} numeric character(s)."

    if feature == "Suspicious word present":
        if value == 1:
            return "A suspicious keyword was detected."
        return "No suspicious keyword was detected."

    if feature == "Number of suspicious words":
        if value == 0:
            return "No suspicious keywords were detected."
        return f"{value} suspicious keyword(s) were detected."

    if feature == "Suspicious TLD":
        if value == 1:
            return "The domain uses a TLD included in the model's suspicious-TLD features."
        return "The domain does not use one of the model's suspicious TLDs."

    if feature == "Multiple subdomains":
        if value == 1:
            return "The domain contains multiple subdomains."
        return "The domain does not contain multiple subdomains."

    if feature == "Hyphen in domain":
        if value == 1:
            return "The domain contains a hyphen."
        return "No hyphen was detected in the domain."

    if feature == "Digit in domain":
        if value == 1:
            return "The domain contains numeric characters."
        return "No numeric characters were detected in the domain."

    if feature == "Domain length":
        if value > 25:
            return f"The domain is relatively long ({value} characters)."
        return f"The domain contains {value} characters."

    if feature == "URL length > 60":
        if value == 1:
            return "The URL exceeds 60 characters."
        return "The URL does not exceed 60 characters."

    return "This feature contributed to the model's prediction."


# Cache the SHAP explainer so it is not recreated for every request.
_explainer = None


def get_explainer():
    global _explainer

    if _explainer is None:
        model = load_model()
        _explainer = shap.TreeExplainer(model)

    return _explainer


def explain_url(url, top_n=5):
    model = load_model()
    features = extract_features(url)

    explainer = get_explainer()

    X = np.array([features], dtype=float)
    shap_values = explainer.shap_values(X)

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

    # Show the features with the strongest absolute influence first.
    feature_impacts.sort(
        key=lambda x: x["impact"],
        reverse=True
    )

    explanations = []

    for item in feature_impacts[:top_n]:
        shap_value = item["shap_value"]

        explanations.append({
            "feature": item["feature"],
            "value": item["value"],
            "impact": round(shap_value, 4),
            "reason": get_feature_reason(
                item["feature"],
                item["value"]
            )
        })

    return explanations