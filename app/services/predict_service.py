def predict_url(url):
    if "login" in url or "verify" in url:
        return {
            "url": url,
            "is_phishing": True,
            "confidence": 0.85,
            "message": "Phishing detected (rule-based)"
        }
    else:
        return {
            "url": url,
            "is_phishing": False,
            "confidence": 0.60,
            "message": "Likely safe"
        }