
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml_model" / "src"))

import train_200k_experiment as exp

FEED_FILE = ROOT / "ml_model" / "external_data" / "openphish_feed.txt"
TRAIN_FILE = ROOT / "ml_model" / "dataset" / "phishing.csv"

MODEL_FILE = (
    ROOT / "ml_model" / "results" / "cnn_candidate_domain_adapt_v1"
    / "character_cnn_candidate.keras"
)

OUT = ROOT / "ml_model" / "results" / "cnn_candidate_openphish_v1"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHISHSHIELD - CANDIDATE CNN OPENPHISH EVALUATION")
print("=" * 60)

for path in [FEED_FILE, TRAIN_FILE, MODEL_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# Load original URLs only for exact-overlap checking.
print("\nLoading original training URLs...")
training = pd.read_csv(TRAIN_FILE, encoding="latin1")
training_urls = set()

for raw in training["URL"].dropna().astype(str):
    try:
        url = exp.normalize_url(raw)
        if url:
            training_urls.add(url)
    except (ValueError, TypeError, AttributeError):
        continue

print("Unique normalized training URLs:", len(training_urls))

# Read and normalize the existing OpenPhish feed.
print("\nReading OpenPhish feed...")
lines = FEED_FILE.read_text(
    encoding="utf-8", errors="replace"
).splitlines()

records = []
seen = set()
invalid = 0
duplicates = 0

for raw in lines:
    raw = raw.strip()
    if not raw:
        continue

    try:
        url = exp.normalize_url(raw)
    except (ValueError, TypeError, AttributeError):
        invalid += 1
        continue

    if not url:
        invalid += 1
        continue

    if url in seen:
        duplicates += 1
        continue

    seen.add(url)
    records.append({
        "original_url": raw,
        "normalized_url": url,
        "overlaps_original_training": url in training_urls,
    })

if not records:
    raise ValueError("No valid URLs found in the OpenPhish feed.")

df = pd.DataFrame(records)
print("Valid unique URLs:", len(df))
print("Invalid lines:", invalid)
print("Duplicate URLs:", duplicates)
print("Exact overlaps with original training data:",
      int(df["overlaps_original_training"].sum()))

# Load the saved candidate CNN; do not retrain it.
print("\nLoading saved candidate CNN...")
model = tf.keras.models.load_model(MODEL_FILE)

print("Encoding URLs with the existing CNN preprocessing...")
X = exp.encode_urls(df["normalized_url"].tolist())

print("Predicting...")
prob = model.predict(X, batch_size=512, verbose=1).ravel()
pred = (prob >= 0.5).astype("int32")

df["phishing_probability"] = prob
df["prediction"] = pred
df["prediction_label"] = df["prediction"].map({
    0: "Legitimate",
    1: "Phishing",
})

# This is a phishing-feed prediction rate, not measured recall:
# the feed does not supply independently verified ground truth
# for every entry, and it has no legitimate comparison group.
total = len(df)
predicted_phishing = int((pred == 1).sum())
predicted_legitimate = int((pred == 0).sum())

nonoverlap = ~df["overlaps_original_training"]
n_nonoverlap = int(nonoverlap.sum())
nonoverlap_phishing = int(
    (df.loc[nonoverlap, "prediction"] == 1).sum()
)

summary = {
    "evaluation_time_utc": datetime.now(timezone.utc).isoformat(),
    "model": str(MODEL_FILE.relative_to(ROOT)),
    "feed": str(FEED_FILE.relative_to(ROOT)),
    "threshold": 0.5,
    "unique_urls_processed": total,
    "predicted_phishing": predicted_phishing,
    "predicted_legitimate": predicted_legitimate,
    "phishing_prediction_rate": predicted_phishing / total,
    "exact_training_overlap_count": int(
        df["overlaps_original_training"].sum()
    ),
    "nonoverlapping_url_count": n_nonoverlap,
    "nonoverlapping_predicted_phishing": nonoverlap_phishing,
    "nonoverlapping_phishing_prediction_rate": (
        nonoverlap_phishing / n_nonoverlap
        if n_nonoverlap else None
    ),
    "invalid_lines_skipped": invalid,
    "duplicate_urls_skipped": duplicates,
    "accuracy": None,
    "precision": None,
    "recall": None,
    "false_positive_rate": None,
    "interpretation": (
        "Phishing-only feed: prediction rates only. "
        "Accuracy, precision, recall and false-positive rate "
        "cannot be measured from this feed alone."
    ),
}

df.to_csv(OUT / "predictions.csv", index=False)

with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame([summary]).to_csv(OUT / "summary.csv", index=False)

print("\nRESULTS")
print(json.dumps(summary, indent=2))
print("\nFiles saved to:", OUT)
print("No model was retrained or replaced.")
