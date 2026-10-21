
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml_model" / "src"))

import train_200k_experiment as exp

MODEL_PATH = (
    ROOT / "ml_model" / "results"
    / "cnn_candidate_domain_adapt_v1"
    / "character_cnn_candidate.keras"
)

# Replace these example URLs with your own 25 URLs.
URLS = [
    "https://www.google.com/",
    "https://www.microsoft.com/",
    "https://www.wikipedia.org/",
    "https://www.amazon.com/",
    "https://www.github.com/",
    "https://www.python.org/",
    "https://www.apple.com/",
    "https://www.bbc.com/",
    "https://www.nytimes.com/",
    "https://www.mozilla.org/",
    "http://secure-login-verify-account.example.com/login",
    "http://paypal-security-check.example.net/verify",
    "http://account-update-confirm.example.org/signin",
    "http://192.0.2.1/login/verify",
    "http://banking-alert-security.example.com/account",
    "http://free-prize-claim.example.net/winner",
    "http://microsoft-account-verify.example.org/login",
    "http://amazon-payment-confirm.example.com/update",
    "http://secure-user-check.example.net/signin",
    "http://login-verify-session.example.org/account",
    "https://www.linkedin.com/",
    "https://www.reddit.com/",
    "https://www.adobe.com/",
    "https://www.nasa.gov/",
    "https://www.un.org/",
]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    normalized = [exp.normalize_url(url) for url in URLS]
    X = exp.encode_urls(normalized)

    probabilities = model.predict(X, batch_size=32, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    results = pd.DataFrame({
        "URL": URLS,
        "Phishing_Probability_Percent": np.round(probabilities * 100, 2),
        "Prediction": np.where(
            predictions == 1, "Phishing", "Legitimate"
        ),
    })

    output = (
        ROOT / "ml_model" / "results"
        / "cnn_25_url_test_results.csv"
    )
    results.to_csv(output, index=False)

    print("\nPHISHSHIELD - 25 URL TEST")
    print("Threshold: 0.50")
    print("=" * 80)

    for _, row in results.iterrows():
        print(f"\nURL: {row['URL']}")
        print(
            f"Phishing probability: "
            f"{row['Phishing_Probability_Percent']:.2f}%"
        )
        print(f"Model prediction: {row['Prediction']}")

    print("\nResults saved to:", output)
    print("\nNote: predictions are model estimates, not proof of safety.")


if __name__ == "__main__":
    main()
