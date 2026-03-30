# 🛡️ PhishShield — Intelligent Phishing Detection System

> A machine learning-based cybersecurity system that detects phishing URLs, emails, and SMS messages with real-time risk classification, explainable AI, and a browser extension.

---

## 📌 Overview

PhishShield analyzes URLs and text messages to predict whether they are **Safe** or **Phishing**, along with confidence scores and explanations.

It follows a modular architecture integrating:
- Machine Learning models
- Flask backend APIs
- Web-based UI
- Chrome extension for real-time detection

---

## 🎯 Objectives

- Detect phishing URLs using machine learning
- Provide **Explainable AI (XAI)** outputs
- Extend detection to **Email and SMS**
- Build a **browser extension** for real-time detection
- Store and display detection history

---

## 🚀 Features

- 🔍 Real-time phishing detection for URLs
- 📩 Email and SMS phishing analysis
- 📊 Confidence score with each prediction
- 🧠 Explainable AI insights (why flagged)
- 🌐 Chrome extension for live browsing protection
- 🗂️ Detection history storage and retrieval
- ⚡ Fast API responses using Flask backend
- 📈 Scalable modular architecture

---

## 🧠 System Architecture

```
┌───────────────────────────────────────────────┐
│ Presentation Layer        | Web UI / Extension │
├───────────────────────────────────────────────┤
│ Backend Layer             | Flask REST API     │
├───────────────────────────────────────────────┤
│ Machine Learning Layer    | Feature + Model    │
├───────────────────────────────────────────────┤
│ Database Layer            | Detection Storage  │
└───────────────────────────────────────────────┘
```

---

## 🔄 Workflow

```
User Input (Web / Extension)
        |
        v
Backend API receives request
        |
        v
Feature extraction
        |
        v
ML model prediction
        |
        v
Result + Confidence + Explanation
        |
        v
Display to user + Store in database
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

Centralized in `shared/feature_extractor.py`

| Feature | Description |
|--------|------------|
| URL Length | Total character count |
| Dot Count | Number of `.` in URL |
| HTTPS Presence | Secure protocol check |
| `@` Symbol | Common phishing indicator |
| Hyphen Count | Number of `-` in domain |
| IP Address | Detects direct IP usage |
| Suspicious Patterns | `//`, redirects |
| Digit Ratio | Proportion of digits |

---

### 🔹 Machine Learning

| Property | Details |
|---------|--------|
| Primary Model | Random Forest Classifier |
| Baseline | Logistic Regression |
| Optional | XGBoost |
| Metric | F1 Score ≥ 0.90 |
| Dataset | 50,000+ URLs |
| Split | 70% / 15% / 15% |
| Labels | 0 = Safe, 1 = Phishing |

---

### 🔹 API Endpoints

#### URL Detection

```
POST /predict
```

**Request**
```json
{
  "url": "http://example.com"
}
```

**Response**
```json
{
  "result": "phishing",
  "confidence": 0.94,
  "reason": "Contains suspicious symbols"
}
```

---

#### Text / Email / SMS Detection

```
POST /predict-text
```

**Request**
```json
{
  "text": "Your account has been suspended"
}
```

**Response**
```json
{
  "result": "phishing",
  "confidence": 0.88
}
```

---

### 🔹 Database Schema

**Table: detections**

| Field | Type | Description |
|------|------|------------|
| id | Integer | Primary Key |
| input_value | String | URL or text |
| input_type | String | url / text |
| result | String | safe / phishing |
| confidence | Float | Prediction score |
| timestamp | DateTime | Detection time |

---

### 🔹 Browser Extension

- Detects phishing URLs in real-time
- Sends active tab URL to backend
- Displays result in popup

**Permissions:**
- `activeTab`
- `scripting`

---

### 🔹 Explainable AI

- Uses SHAP / rule-based explanations
- Highlights important features influencing prediction
- Integrated into UI and extension

---

## 🛡️ Error Handling

| Scenario | Behavior |
|---------|----------|
| Invalid URL | Returns error |
| Empty input | Validation error |
| Low confidence | Marked as "Uncertain" |
| Backend failure | Fallback response |

---

## 🛠️ Tech Stack

- **Machine Learning:** Python, scikit-learn, pandas, numpy, SHAP
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Extension:** Chrome Extension (Manifest V3)
- **Database:** SQLite
- **Deployment:** Render / Railway (optional)

---

## 👥 Team

| Role | Responsibilities |
|------|----------------|
| ML Developer | Model training, feature engineering |
| Backend Developer | API development, DB integration |
| Frontend Developer | UI + Extension development |

---

## ▶️ Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/your-username/phishshield.git
cd phishshield
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model
```bash
python ml_model/src/train_url_model.py
```

### 4. Run Backend
```bash
python backend/app.py
```

### 5. Load Chrome Extension

- Open `chrome://extensions/`
- Enable **Developer Mode**
- Click **Load Unpacked**
- Select the `extension/` folder

---

## 🏆 Conclusion

PhishShield is a scalable and modular phishing detection system combining machine learning, explainability, and real-time browser integration.

It is designed as a practical, real-world cybersecurity solution.
