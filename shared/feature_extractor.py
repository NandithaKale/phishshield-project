from urllib.parse import urlparse

from shared.url_normalizer import normalize_url
from app.services.security_checks import analyze_url_security


def extract_features(url):
    """
    Extract the model features from the same normalized URL used by the
    backend security layer.

    The final two features are deterministic security signals:
      18. direct IP address
      19. look-alike domain
    """
    url = normalize_url(url).lower()
    parsed = urlparse(url)
    domain = parsed.hostname or ""

    suspicious_words = [
        "login", "secure", "verify", "account",
        "bank", "update", "free", "bonus", "paypal",
        "signin", "confirm", "security", "alert"
    ]

    suspicious_tlds = [
        ".xyz", ".tk", ".ml", ".ga", ".cf",
        ".gq", ".top", ".biz", ".info"
    ]

    security = analyze_url_security(url)

    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('/'),
        url.count('='),

        int(url.startswith("https")),
        int(url.startswith("http") and not url.startswith("https")),

        int('@' in url),
        sum(c.isdigit() for c in url),

        int(any(word in url for word in suspicious_words)),
        sum(word in url for word in suspicious_words),

        int(any(tld in url for tld in suspicious_tlds)),

        int(domain.count('.') > 2),
        int('-' in domain),
        int(any(c.isdigit() for c in domain)),

        len(domain),

        int(len(url) > 60),

        int(security["is_ip_address"]),
        int(security["lookalike_domain"]["is_lookalike"]),
    ]
