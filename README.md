# 🛡️ PhishShield — Intelligent Phishing Detection System

> A machine learning-based cybersecurity system that detects phishing URLs, emails, and SMS messages with real-time risk classification, explainable AI, and a browser extension.

---

## 📌 Overview

PhishShield analyzes URLs and text messages to predict whether they are **Safe** or **Phishing**, complete with confidence scores and explanations. Built with a modular architecture, it covers everything from a Flask backend and ML model to a Chrome extension for real-time browser-based detection.

---

## 🎯 Objectives

- Detect phishing URLs using machine learning
- Provide **Explainable AI** outputs (why a URL is flagged)
- Extend detection to **email and SMS** messages
- Build a **browser extension** for real-time detection
- Store and display detection history

---

## 🧠 System Architecture

```
┌─────────────────────────────────┐
│       Presentation Layer        │  Web UI / Browser Extension
├─────────────────────────────────┤
│          Backend Layer          │  Flask REST API
├─────────────────────────────────┤
│    Machine Learning Layer       │  Feature Extraction + Random Forest
├─────────────────────────────────┤
│          Database Layer         │  Detection History Storage
└─────────────────────────────────┘
```

### 🔄 Request Workflow

```
User Input (Web / Extension)
        ↓
  Backend API receives request
        ↓
  Feature extraction
        ↓
  ML model predicts probability
        ↓
  Result + Confidence + Explanation
        ↓
  Display to user & store in DB
```

---

## 📂 Project Structure

```
phishshield/
│
├── backend/
│   ├── app.py
│   ├── routes/
│   │   ├── predict_url.py
│   │   └── predict_text.py
│   ├── services/
│   │   ├── predictor.py
│   │   └── model_loader.py
│   └── config.py
│
├── ml_model/
│   ├── dataset/
│   ├── src/
│   │   ├── train_url_model.py
│   │   ├── train_text_model.py
│   │   └── feature_extractor.py
│   └── saved_model/
│       ├── url_model.pkl
│       └── text_model.pkl
│
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   ├── result.html
│   │   └── history.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js
│
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── style.css
│
├── database/
│   ├── db.py
│   └── schema.sql
│
├── shared/
│   └── feature_extractor.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Core Modules

### 🔹 Feature Extraction
All feature logic is centralized in `shared/feature_extractor.py` and extracts:

| Feature | Description |
|---|---|
| URL length | Total character count |
| Dot count | Number of `.` in URL |
| HTTPS presence | Secure protocol check |
| `@` symbol | Common phishing indicator |
| Hyphen count | Number of `-` in domain |
| IP address | Direct IP usage detection |
| Suspicious patterns | `//`, redirects, etc. |
| Digit ratio | Proportion of digits in URL |

### 🔹 Machine Learning

| Property | Detail |
|---|---|
| Primary Model | Random Forest Classifier |
| Baseline | Logistic Regression |
| Optional | XGBoost |
| Primary Metric | F1 Score (target ≥ 0.90) |
| Dataset Size | 50,000+ URLs |
| Data Split | 70% Train / 15% Val / 15% Test |
| Labels | `0` = Safe, `1` = Phishing |

### 🔹 API Endpoints

**URL Detection**
```
POST /predict
```
```json
// Request
{ "url": "http://example.com" }

// Response
{ "result": "phishing", "confidence": 0.94, "reason": "Contains suspicious symbols" }
```

**Text / SMS / Email Detection**
```
POST /predict-text
```
```json
// Request
{ "text": "Your account has been suspended" }

// Response
{ "result": "phishing", "confidence": 0.88 }
```

### 🔹 Database Schema

Table: `detections`

| Field | Type | Description |
|---|---|---|
| `id` | Primary Key | Auto-increment |
| `input_value` | String | URL or text input |
| `input_type` | String | `url` or `text` |
| `result` | String | `safe` or `phishing` |
| `confidence` | Float | Model confidence score |
| `timestamp` | DateTime | Time of detection |

### 🔹 Browser Extension

- **Trigger:** Button click on current page
- **Flow:** Captures active tab URL → sends to backend → displays result in popup
- **Permissions required:** `activeTab`, `scripting`

### 🔹 Explainable AI

- Uses **SHAP** or rule-based explanation
- Returns top features influencing each prediction
- Displayed in both the web UI and extension popup

---

## 🛡️ Error Handling

| Scenario | Behavior |
|---|---|
| Invalid URL | Returns error message |
| Empty input | Validation error |
| Low confidence | Marked as "Uncertain" |
| Backend failure | Fallback message shown |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)

- **ML:** Python, scikit-learn, pandas, numpy, SHAP
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Extension:** Chrome Extension API (Manifest V3)
- **Database:** SQLite / configurable
- **Deployment:** Flask dev server → Render / Railway *(optional)*

---

## 👥 Team

| Role | Responsibilities |
|---|---|
| ML Developer | Dataset preprocessing, feature engineering, model training & evaluation |
| Backend Developer | Flask API development, DB integration, model serving |
| Frontend Developer | UI design, API integration, Chrome extension |

---

## ▶️ Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/your-username/phishshield.git
cd phishshield
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python ml_model/src/train_url_model.py
```

**4. Start the backend**
```bash
python backend/app.py
```

**5. Load the Chrome extension**
- Open `chrome://extensions/`
- Enable **Developer Mode**
- Click **Load Unpacked** → select the `extension/` folder

---

## 🏆 Conclusion

PhishShield is a scalable and modular phishing detection system with a clean separation of concerns across ML, backend, frontend, and browser layers. It combines strong model performance with explainability and multi-input support to deliver a practical, real-world cybersecurity tool.

---

> ⭐ Star this repo if you found it useful!
