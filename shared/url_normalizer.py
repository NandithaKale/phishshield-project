"""URL normalization shared by training, prediction and XAI."""
from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    """
    Canonicalize a user-supplied URL without resolving/fetching it.

    - strips surrounding whitespace
    - adds https:// when no scheme is supplied
    - lowercases the hostname
    - preserves path/query
    - removes the fragment because it is not sent to the server
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    value = url.strip()
    if not value:
        raise ValueError("URL is required")

    if "://" not in value:
        value = "https://" + value

    parsed = urlsplit(value)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported")

    if not parsed.hostname:
        raise ValueError("Invalid URL: hostname is missing")

    hostname = parsed.hostname.lower().rstrip(".")
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        raise ValueError("Invalid internationalized hostname")

    # Rebuild netloc while retaining an explicit port and credentials.
    # Userinfo is preserved because it can itself be a phishing signal.
    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, _ = netloc.rsplit("@", 1)
        netloc = f"{userinfo}@{hostname}"
    else:
        netloc = hostname

    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    return urlunsplit((
        parsed.scheme.lower(),
        netloc,
        parsed.path or "",
        parsed.query,
        "",  # fragment removed
    ))
