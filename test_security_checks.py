from shared.url_normalizer import normalize_url
from app.services.security_checks import analyze_url_security


TEST_CASES = [
    " google.com ",
    "https://www.google.com/login",
    "http://192.168.1.10/login",
    "https://paypa1.com/login",
    "https://g00gle.com/account",
    "https://github.com/user/repository",
    "example.com#tracking-fragment",
]


def main():
    print("=== URL NORMALIZATION + SECURITY CHECKS ===\n")

    for raw_url in TEST_CASES:
        try:
            normalized = normalize_url(raw_url)
            security = analyze_url_security(normalized)

            print("INPUT      :", raw_url)
            print("NORMALIZED :", normalized)
            print("IP ADDRESS :", security["is_ip_address"])
            print(
                "LOOKALIKE  :",
                security["lookalike_domain"]["is_lookalike"],
                security["lookalike_domain"]["matched_domain"],
                security["lookalike_domain"]["similarity"],
            )
            print("RISK FLAGS :", security["risk_flags"])
            print("-" * 70)
        except ValueError as exc:
            print("INPUT      :", raw_url)
            print("ERROR      :", exc)
            print("-" * 70)


if __name__ == "__main__":
    main()
