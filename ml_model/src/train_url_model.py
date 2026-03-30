import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from shared.feature_extractor import extract_features

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("ml_model/dataset/phishing.csv", encoding='latin1')
data = data.dropna()

# =========================
# FIX LABELS
# =========================
data['Label'] = data['Label'].astype(str).str.lower()
data['Label'] = data['Label'].map({'good': 0, 'bad': 1})
data = data.dropna(subset=['Label'])
data['Label'] = data['Label'].astype(int)

print("\nLabel Count:\n", data['Label'].value_counts())

# =========================
# BALANCE DATASET
# =========================
df_good = data[data['Label'] == 0]
df_bad = data[data['Label'] == 1]

df_bad_upsampled = resample(
    df_bad,
    replace=True,
    n_samples=len(df_good),
    random_state=42
)

data = pd.concat([df_good, df_bad_upsampled])

# =========================
# FEATURES
# =========================
X = []
y = data['Label']

for url in data['URL']:
    X.append(extract_features(str(url)))

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# SAVE MODEL
pickle.dump(model, open("ml_model/saved_model/url_model.pkl", "wb"))

# =========================
# PREDICTION FUNCTION
# =========================
def predict_url_with_risk(url):
    features = extract_features(url)
    proba = model.predict_proba([features])[0]

    phishing_prob = proba[1] * 100

    if phishing_prob < 60:
        phishing_prob *= 0.6

    if phishing_prob < 30:
        label = "Safe"
    elif phishing_prob < 70:
        label = "Moderate Risk"
    else:
        label = "Phishing"

    return label, round(phishing_prob, 2)

# =========================
# TEST URLS
# =========================
test_urls = [
# SAFE (15)
"https://www.google.com",
"https://www.amazon.in",
"https://www.microsoft.com",
"https://www.github.com",
"https://www.stackoverflow.com",
"https://www.wikipedia.org",
"https://www.apple.com",
"https://www.netflix.com",
"https://www.linkedin.com",
"https://www.paypal.com",
"https://www.ibm.com",
"https://www.oracle.com",
"https://www.adobe.com",
"https://www.nvidia.com",
"https://www.intel.com",

# PHISHING (15)
"http://secure-login-paypal.xyz/verify",
"http://account-update-bank.ru/login",
"http://free-money-offer.click/win",
"http://verify-your-account-now.info",
"http://login-microsoft-account-alert.xyz",
"http://paypal-security-check.ga",
"http://update-your-bank-account-now.top",
"http://confirm-identity-paypal-alert.ml",
"http://secure-amazon-login-warning.xyz",
"http://urgent-bank-verification-alert.biz",
"http://signin-ebay-account-security.xyz",
"http://free-gift-card-amazon.click",
"http://verify-your-login-now-security.tk",
"http://bank-secure-update-account.xyz/login",
"http://amazon-security-alert-confirm.ga"
]

# ACTUAL LABELS
actual_labels = [0]*15 + [1]*15

predicted_labels = []
results_table = []

print("\nModel Accuracy:", round(accuracy, 4))
print("\n--- Testing Results ---")

for i, url in enumerate(test_urls):
    label, confidence = predict_url_with_risk(url)

    pred = 1 if label == "Phishing" else 0
    predicted_labels.append(pred)

    actual = actual_labels[i]

    results_table.append({
        "URL": url,
        "Actual": "Phishing" if actual == 1 else "Safe",
        "Predicted": label,
        "Confidence (%)": confidence
    })

    print(url, "→", label, f"({confidence}%)")

# =========================
# TABLE
# =========================
df_results = pd.DataFrame(results_table)

print("\n--- Comparison Table ---")
print(df_results)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
f1 = f1_score(actual_labels, predicted_labels)

print("\n--- Evaluation Metrics ---")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))