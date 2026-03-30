# 🛡️ PhishShield — Intelligent Phishing Detection System

> A machine learning-based cybersecurity system that detects phishing URLs with real-time risk classification and confidence scoring.

---

## 📌 Overview

PhishShield analyzes structural and lexical features of URLs to predict whether they are **Safe**, **Moderate Risk**, or **Phishing**. Built with a modular and scalable architecture, it is designed to extend into a full real-world application with a backend API, browser extension, and multi-platform phishing detection.

---

## 🎯 Objectives

- Detect phishing URLs using machine learning
- Provide risk-based classification with confidence scores
- Build a scalable system (ML + Backend + Frontend)
- Extend detection to a browser extension for real-time use
- Provide explainability for model predictions *(future scope)*

---

## 🧠 System Architecture

PhishShield follows a layered architecture:

```
┌─────────────────────────────────┐
│       Presentation Layer        │  Web UI / Browser Extension
├─────────────────────────────────┤
│          API Layer              │  Flask Backend
├─────────────────────────────────┤
│    Machine Learning Layer       │  Feature Extraction + Random Forest
├─────────────────────────────────┤
│          Data Layer             │  URL History & Results (planned)
└─────────────────────────────────┘
```

### 🔄 Request Workflow

```
User Input (URL)
      ↓
Backend receives request
      ↓
Feature extraction
      ↓
ML model predicts probability
      ↓
Risk classification applied
      ↓
Result returned with confidence score
      ↓
(Optional) Data stored for history
```

---

## 📂 Project Structure

```
phishshield_project/
│
├── ml_model/
│   ├── dataset/
│   │   └── phishing_dataset.csv
│   ├── src/
│   │   ├── train_url_model.py
│   │   ├── predict_url.py
│   │   └── model_utils.py
│   └── saved_model/
│       └── url_model.pkl
│
├── backend/
│   ├── app.py
│   ├── routes/
│   │   ├── url_routes.py
│   │   ├── email_routes.py
│   │   └── sms_routes.py
│   ├── services/
│   │   ├── prediction_service.py
│   │   ├── explainability_service.py
│   │   └── history_service.py
│   └── database/
│       └── db_config.py
│
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   ├── result.html
│   │   └── dashboard.html
│   └── static/
│       ├── css/
│       └── js/
│
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── background.js
│
├── shared/
│   ├── feature_extractor.py
│   ├── utils.py
│   └── constants.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Core Modules

### 🔹 Feature Extraction
Extracts key URL attributes including:
- URL length
- Number of dots, digits, and special characters
- Presence of HTTPS
- Suspicious keywords (e.g., `login`, `verify`, `bank`)
- Suspicious TLDs (e.g., `.xyz`, `.tk`, `.ml`)

### 🔹 Machine Learning
| Property | Detail |
|---|---|
| Algorithm | Random Forest Classifier |
| Input | Extracted URL features |
| Output | Phishing probability score |

### 🔹 Risk Classification
| Confidence Score | Classification |
|---|---|
| 0% – 30% | ✅ Safe |
| 30% – 70% | ⚠️ Moderate Risk |
| 70% – 100% | 🚨 Phishing |

### 🔹 Backend API *(Planned)*
Flask-based REST API connecting the frontend to the ML model.

### 🔹 Browser Extension *(Planned)*
Detects the current page URL, sends it to the backend, and displays a real-time warning.

### 🔹 Explainability Module *(Future)*
Highlights which URL features triggered the phishing classification.

---

## 🤖 ML Workflow

```
Training Phase                  Prediction Phase
──────────────                  ────────────────
Load dataset          →         Input URL
Extract features      →         Extract features
Train Random Forest   →         Load saved model (.pkl)
Save model (.pkl)     →         Predict probability
                                Convert to risk level
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy (Training) | ~84% |
| Accuracy (Test Cases) | ~96.67% |
| Precision | 1.0 |
| Recall | 0.93 |
| F1 Score | 0.96 |

---

## 🚀 Features

| Status | Feature |
|---|---|
| ✅ Done | URL phishing detection |
| ✅ Done | Feature engineering |
| ✅ Done | ML model training & prediction |
| ✅ Done | Confidence score output |
| ✅ Done | Risk-based classification |
| 🔄 In Progress | Flask backend API |
| 🔄 In Progress | Frontend UI integration |
| 🔮 Planned | Browser extension |
| 🔮 Planned | Email / SMS phishing detection |
| 🔮 Planned | Explainable AI |
| 🔮 Planned | Analytics dashboard |
| 🔮 Planned | Database integration |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

- **ML:** Python, scikit-learn, pandas, numpy
- **Backend:** Flask *(planned)*
- **Frontend:** HTML, CSS, JavaScript *(planned)*

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

**4. Run a prediction**
```bash
python ml_model/src/predict_url.py
```

---

## 👥 Team

| Role | Responsibilities |
|---|---|
| ML Developer | Feature engineering, model training, evaluation |
| Backend Developer | API development and integration |
| Frontend Developer | UI design and user interaction |

---

## 🏆 Conclusion

PhishShield is a scalable and intelligent phishing detection system that leverages machine learning to identify malicious URLs. The current system demonstrates strong performance and provides a solid foundation for building a complete, real-world cybersecurity product.

---

> ⭐ Star this repo if you found it useful!
