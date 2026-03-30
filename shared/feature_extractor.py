from urllib.parse import urlparse

def extract_features(url):
    url = url.lower()
    parsed = urlparse(url)
    domain = parsed.netloc

    suspicious_words = [
        "login", "secure", "verify", "account",
        "bank", "update", "free", "bonus", "paypal",
        "signin", "confirm", "security", "alert"
    ]

    suspicious_tlds = [
        ".xyz", ".tk", ".ml", ".ga", ".cf",
        ".gq", ".top", ".biz", ".info"
    ]

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
    ]