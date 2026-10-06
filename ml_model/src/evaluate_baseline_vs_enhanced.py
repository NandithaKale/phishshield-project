"""
Baseline vs enhanced evaluation.

Run from the repository root:
    python ml_model/src/evaluate_baseline_vs_enhanced.py

Optional quick run:
    python ml_model/src/evaluate_baseline_vs_enhanced.py --sample 50000
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# Allow imports from the repository root
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)

from shared.feature_extractor import extract_features


def baseline_extract_features(url):
    """
    The original 17-feature extractor,
    preserved for a fair baseline comparison.
    """

    from urllib.parse import urlparse

    url = str(url).lower()

    parsed = urlparse(url)
    domain = parsed.netloc

    suspicious_words = [
        "login",
        "secure",
        "verify",
        "account",
        "bank",
        "update",
        "free",
        "bonus",
        "paypal",
        "signin",
        "confirm",
        "security",
        "alert",
    ]

    suspicious_tlds = [
        ".xyz",
        ".tk",
        ".ml",
        ".ga",
        ".cf",
        ".gq",
        ".top",
        ".biz",
        ".info",
    ]

    return [
        # 1. URL length
        len(url),

        # 2. Number of dots
        url.count("."),

        # 3. Number of hyphens
        url.count("-"),

        # 4. Number of slashes
        url.count("/"),

        # 5. Number of equal signs
        url.count("="),

        # 6. HTTPS
        int(url.startswith("https")),

        # 7. HTTP without HTTPS
        int(
            url.startswith("http")
            and not url.startswith("https")
        ),

        # 8. @ symbol
        int("@" in url),

        # 9. Number of digits
        sum(c.isdigit() for c in url),

        # 10. Suspicious word
        int(
            any(
                word in url
                for word in suspicious_words
            )
        ),

        # 11. Suspicious word count
        sum(
            word in url
            for word in suspicious_words
        ),

        # 12. Suspicious TLD
        int(
            any(
                tld in url
                for tld in suspicious_tlds
            )
        ),

        # 13. Multiple subdomains
        int(domain.count(".") > 2),

        # 14. Hyphen in domain
        int("-" in domain),

        # 15. Digit in domain
        int(
            any(
                c.isdigit()
                for c in domain
            )
        ),

        # 16. Domain length
        len(domain),

        # 17. URL length > 60
        int(len(url) > 60),
    ]


def build_model():
    """
    Creates the Random Forest classifier.
    """

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
    }


def main():

    # ==========================================================
    # 1. Parse command-line arguments
    # ==========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Optional number of rows to use "
            "before balancing for a quicker run."
        ),
    )

    args = parser.parse_args()


    # ==========================================================
    # 2. Load dataset
    # ==========================================================

    data = (
        pd.read_csv(
            "ml_model/dataset/phishing.csv",
            encoding="latin1",
        )
        .dropna()
    )


    # ==========================================================
    # 3. Convert labels
    # ==========================================================

    data["Label"] = (
        data["Label"]
        .astype(str)
        .str.lower()
        .map(
            {
                "good": 0,
                "bad": 1,
            }
        )
    )

    data = data.dropna(
        subset=["Label"]
    )

    data["Label"] = data["Label"].astype(int)


    # ==========================================================
    # 4. Optional sampling
    # ==========================================================

    if args.sample:

        data = data.sample(
            n=min(
                args.sample,
                len(data),
            ),
            random_state=42,
        )


    # ==========================================================
    # 5. Balance the classes
    # ==========================================================

    good = data[
        data["Label"] == 0
    ]

    bad = data[
        data["Label"] == 1
    ]

    bad = resample(
        bad,
        replace=True,
        n_samples=len(good),
        random_state=42,
    )

    data = pd.concat(
        [
            good,
            bad,
        ],
        ignore_index=True,
    )


    # ==========================================================
    # 6. Get URLs and labels
    # ==========================================================

    urls = (
        data["URL"]
        .astype(str)
        .tolist()
    )

    y = data[
        "Label"
    ].to_numpy()


    # ==========================================================
    # 7. Validate URLs
    #
    # The enhanced extractor uses normalize_url().
    # Some dataset rows may contain malformed URLs.
    #
    # We validate BEFORE splitting the dataset so that
    # baseline and enhanced models use exactly the same
    # URLs and labels.
    # ==========================================================

    valid_urls = []
    valid_labels = []

    invalid_count = 0

    print(
        "Checking URLs for invalid/malformed entries..."
    )

    for url, label in zip(
        urls,
        y,
    ):

        try:

            # The enhanced extractor performs
            # URL normalization and validation.
            extract_features(url)

            valid_urls.append(url)
            valid_labels.append(label)

        except (
            ValueError,
            TypeError,
            AttributeError,
        ) as e:

            invalid_count += 1

            print(
                f"Skipping invalid URL: {repr(url)}"
            )

            print(
                f"Reason: {e}"
            )


    # Replace original data with validated data
    urls = valid_urls

    y = np.array(
        valid_labels,
        dtype=int,
    )


    print()
    print(
        f"Valid URLs: {len(urls)}"
    )

    print(
        f"Removed invalid URLs: {invalid_count}"
    )


    # ==========================================================
    # 8. Make sure enough data remains
    # ==========================================================

    if len(urls) == 0:

        raise ValueError(
            "No valid URLs remain after URL validation."
        )

    if len(np.unique(y)) < 2:

        raise ValueError(
            "Only one class remains after URL validation. "
            "Both good and bad URLs are required."
        )


    # ==========================================================
    # 9. One shared train/test split
    #
    # This is important for a fair comparison.
    #
    # Both baseline and enhanced models use exactly
    # the same training and testing URLs.
    # ==========================================================

    train_idx, test_idx = train_test_split(
        np.arange(
            len(urls)
        ),

        test_size=0.20,

        stratify=y,

        random_state=42,
    )


    # ==========================================================
    # 10. Extract baseline features
    # ==========================================================

    print()
    print(
        "Extracting baseline features..."
    )

    X_baseline = [
        baseline_extract_features(url)
        for url in urls
    ]


    # ==========================================================
    # 11. Extract enhanced features
    # ==========================================================

    print(
        "Extracting enhanced features..."
    )

    X_enhanced = [
        extract_features(url)
        for url in urls
    ]


    # ==========================================================
    # 12. Convert to NumPy arrays
    # ==========================================================

    X_baseline = np.array(
        X_baseline,
        dtype=float,
    )

    X_enhanced = np.array(
        X_enhanced,
        dtype=float,
    )


    # ==========================================================
    # 13. Display feature counts
    # ==========================================================

    print()
    print(
        f"Baseline feature count: "
        f"{X_baseline.shape[1]}"
    )

    print(
        f"Enhanced feature count: "
        f"{X_enhanced.shape[1]}"
    )


    # ==========================================================
    # 14. Build models
    # ==========================================================

    print()
    print(
        "Building baseline model..."
    )

    baseline = build_model()

    print(
        "Building enhanced model..."
    )

    enhanced = build_model()


    # ==========================================================
    # 15. Train baseline model
    # ==========================================================

    print()
    print(
        "Training baseline model..."
    )

    baseline.fit(
        X_baseline[train_idx],
        y[train_idx],
    )


    # ==========================================================
    # 16. Train enhanced model
    # ==========================================================

    print(
        "Training enhanced model..."
    )

    enhanced.fit(
        X_enhanced[train_idx],
        y[train_idx],
    )


    # ==========================================================
    # 17. Baseline predictions
    # ==========================================================

    print()
    print(
        "Generating baseline predictions..."
    )

    baseline_pred = baseline.predict(
        X_baseline[test_idx]
    )


    # ==========================================================
    # 18. Enhanced predictions
    # ==========================================================

    print(
        "Generating enhanced predictions..."
    )

    enhanced_pred = enhanced.predict(
        X_enhanced[test_idx]
    )


    # ==========================================================
    # 19. Calculate metrics
    # ==========================================================

    baseline_metrics = metrics(
        y[test_idx],
        baseline_pred,
    )

    enhanced_metrics = metrics(
        y[test_idx],
        enhanced_pred,
    )


    # ==========================================================
    # 20. Display baseline results
    # ==========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "=== BASELINE (17 features) ==="
    )

    print(
        "=========================================="
    )

    for key, value in baseline_metrics.items():

        print(
            f"{key}: {value}"
        )


    # ==========================================================
    # 21. Display enhanced results
    # ==========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "=== ENHANCED (19 features + "
        "normalization/security signals) ==="
    )

    print(
        "=========================================="
    )

    for key, value in enhanced_metrics.items():

        print(
            f"{key}: {value}"
        )


    # ==========================================================
    # 22. Compare results
    # ==========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "=== CHANGE (enhanced - baseline) ==="
    )

    print(
        "=========================================="
    )

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
    ):

        change = (
            enhanced_metrics[key]
            - baseline_metrics[key]
        )

        print(
            f"{key}: {change:+.4f}"
        )


    # ==========================================================
    # 23. Save enhanced model
    # ==========================================================

    save_path = (
        "ml_model/saved_model/url_model.pkl"
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    import joblib

    joblib.dump(
        enhanced,
        save_path,
    )

    print()
    print(
        "Enhanced model saved to:"
    )

    print(
        save_path
    )


# ==============================================================
# Entry point
# ==============================================================

if __name__ == "__main__":
    main()