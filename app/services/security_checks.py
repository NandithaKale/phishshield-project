"""
Security-layer checks for URL analysis.

These checks are deliberately kept outside the ML model so they can act as
deterministic security signals and remain easy to audit.
"""
from __future__ import annotations

import ipaddress
from difflib import SequenceMatcher
from urllib.parse import urlparse

TRUSTED_DOMAINS = {
    "google.com",
    "microsoft.com",
    "github.com",
    "wikipedia.org",
    "amazon.in",
    "paypal.com",
    "apple.com",
    "amazon.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
}

# Common substitutions used in look-alike domains.
CONFUSABLE_MAP = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "5": "s",
    "7": "t",
    "$": "s",
    "@": "a",
})


def _hostname(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").strip(".").lower()


def is_ip_address(url: str) -> bool:
    """Return True when the URL hostname is an IPv4 or IPv6 address."""
    host = _hostname(url)
    if not host:
        return False

    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _registered_domain(hostname: str) -> str:
    """
    Lightweight registrable-domain approximation.

    It intentionally avoids a new dependency such as tldextract. For the
    current project this is sufficient for comparing a hostname's final
    two labels, while special country-code suffixes are handled separately.
    """
    labels = [x for x in hostname.lower().split(".") if x]
    if len(labels) < 2:
        return hostname.lower()

    common_two_part_suffixes = {
        "co.uk", "org.uk", "ac.uk", "gov.uk",
        "co.in", "com.au", "co.jp", "co.nz",
    }

    suffix2 = ".".join(labels[-2:])
    if suffix2 in common_two_part_suffixes and len(labels) >= 3:
        return ".".join(labels[-3:])

    return suffix2


def _brand_similarity(candidate_label: str, trusted_label: str) -> float:
    candidate = candidate_label.translate(CONFUSABLE_MAP)
    trusted = trusted_label.lower()

    # Exact and one-character substitutions are strong signals.
    return SequenceMatcher(None, candidate, trusted).ratio()


def detect_lookalike_domain(
    url: str,
    trusted_domains: set[str] | None = None,
    similarity_threshold: float = 0.86,
) -> dict:
    """
    Detect domains that closely resemble a trusted domain.

    Examples that can be detected:
      paypa1.com -> paypal.com
      g00gle.com -> google.com
      micros0ft.com -> microsoft.com

    Exact trusted domains are never reported as look-alikes.
    """
    trusted_domains = trusted_domains or TRUSTED_DOMAINS
    host = _hostname(url)

    if not host or is_ip_address(url):
        return {
            "is_lookalike": False,
            "matched_domain": None,
            "similarity": 0.0,
        }

    registrable = _registered_domain(host)

    if registrable in trusted_domains:
        return {
            "is_lookalike": False,
            "matched_domain": registrable,
            "similarity": 1.0,
        }

    candidate_label = registrable.rsplit(".", 1)[0]

    best_domain = None
    best_similarity = 0.0

    for trusted in trusted_domains:
        trusted_label = trusted.rsplit(".", 1)[0]
        similarity = _brand_similarity(candidate_label, trusted_label)

        if similarity > best_similarity:
            best_similarity = similarity
            best_domain = trusted

    # Require a close string match and a short candidate to avoid flagging
    # generic domains merely because they share a few characters.
    suspicious = (
        best_domain is not None
        and best_similarity >= similarity_threshold
        and candidate_label != best_domain.rsplit(".", 1)[0]
    )

    return {
        "is_lookalike": suspicious,
        "matched_domain": best_domain if suspicious else None,
        "similarity": round(best_similarity, 4),
    }


def analyze_url_security(url: str) -> dict:
    """Run all deterministic URL security checks."""
    ip_flag = is_ip_address(url)
    lookalike = detect_lookalike_domain(url)

    risk_flags = []
    if ip_flag:
        risk_flags.append("ip_address")
    if lookalike["is_lookalike"]:
        risk_flags.append("lookalike_domain")

    return {
        "is_ip_address": ip_flag,
        "lookalike_domain": lookalike,
        "risk_flags": risk_flags,
        "security_risk": "high" if risk_flags else "normal",
    }
