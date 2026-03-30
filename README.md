dont use phases in this9:27 PMRan a command, created a file, read a fileRan a command, created a file, read a fileDone! The README now covers everything from the planning doc — architecture, modules, API design, database schema, extension, error handling, and team roles — with no mention of phases. Just replace your-username in the clone URL before pushing to GitHub.ReadmeDocument · MD Downloadhey i want you to like remove feature statue9:29 PMEdited a file, read a fileEdited a file, read a fileDone! The features status table has been removed. Everything else stays the same.ReadmeDocument · MD Downloadgive here only n i hope you retain the features9:29 PMHere's the updated README:
🛡️ PhishShield — Intelligent Phishing Detection System

A machine learning-based cybersecurity system that detects phishing URLs, emails, and SMS messages with real-time risk classification, explainable AI, and a browser extension.


📌 Overview
PhishShield analyzes URLs and text messages to predict whether they are Safe or Phishing, complete with confidence scores and explanations. Built with a modular architecture, it covers everything from a Flask backend and ML model to a Chrome extension for real-time browser-based detection.

🎯 Objectives

Detect phishing URLs using machine learning
Provide Explainable AI outputs (why a URL is flagged)
Extend detection to email and SMS messages
Build a browser extension for real-time detection
Store and display detection history


🧠 System Architecture
┌─────────────────────────────────┐
│       Presentation Layer        │  Web UI / Browser Extension
├─────────────────────────────────┤
│          Backend Layer          │  Flask REST API
├─────────────────────────────────┤
│    Machine Learning Layer       │  Feature Extraction + Random Forest
├─────────────────────────────────┤
│          Database Layer         │  Detection History Storage
└─────────────────────────────────┘
🔄 Request Workflow
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

📂 Project Structure
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

⚙️ Core Modules
🔹 Feature Extraction
All feature logic is centralized in shared/feature_extractor.py and extracts:
FeatureDescriptionURL lengthTotal character countDot countNumber of . in URLHTTPS presenceSecure protocol check@ symbolCommon phishing indicatorHyphen countNumber of - in domainIP addressDirect IP usage detectionSuspicious patterns//, redirects, etc.Digit ratioProportion of digits in URL
🔹 Machine Learning
PropertyDetailPrimary ModelRandom Forest ClassifierBaselineLogistic RegressionOptionalXGBoostPrimary MetricF1 Score (target ≥ 0.90)Dataset Size50,000+ URLsData Split70% Train / 15% Val / 15% TestLabels0 = Safe, 1 = Phishing
🔹 API Endpoints
URL Detection
POST /predict
json{ "url": "http://example.com" }
{ "result": "phishing", "confidence": 0.94, "reason": "Contains suspicious symbols" }
```

**Text / SMS / Email Detection**
```
POST /predict-text
json{ "text": "Your account has been suspended" }
{ "result": "phishing", "confidence": 0.88 }
🔹 Database Schema
Table: detections
FieldTypeDescriptionidPrimary KeyAuto-incrementinput_valueStringURL or text inputinput_typeStringurl or textresultStringsafe or phishingconfidenceFloatModel confidence scoretimestampDateTimeTime of detection
🔹 Browser Extension

Trigger: Button click on current page
Flow: Captures active tab URL → sends to backend → displays result in popup
Permissions required: activeTab, scripting

🔹 Explainable AI

Uses SHAP or rule-based explanation
Returns top features influencing each prediction
Displayed in both the web UI and extension popup


🛡️ Error Handling
ScenarioBehaviorInvalid URLReturns error messageEmpty inputValidation errorLow confidenceMarked as "Uncertain"Backend failureFallback message shown

🛠️ Tech Stack
Show Image
Show Image
Show Image
Show Image
Show Image
Show Image

ML: Python, scikit-learn, pandas, numpy, SHAP
Backend: Flask
Frontend: HTML, CSS, JavaScript
Extension: Chrome Extension API (Manifest V3)
Database: SQLite / configurable
Deployment: Flask dev server → Render / Railway (optional)


👥 Team
RoleResponsibilitiesML DeveloperDataset preprocessing, feature engineering, model training & evaluationBackend DeveloperFlask API development, DB integration, model servingFrontend DeveloperUI design, API integration, Chrome extension

▶️ Getting Started
1. Clone the repository
bashgit clone https://github.com/your-username/phishshield.git
cd phishshield
2. Install dependencies
bashpip install -r requirements.txt
3. Train the model
bashpython ml_model/src/train_url_model.py
4. Start the backend
bashpython backend/app.py
5. Load the Chrome extension

Open chrome://extensions/
Enable Developer Mode
Click Load Unpacked → select the extension/ folder


🏆 Conclusion
PhishShield is a scalable and modular phishing detection system with a clean separation of concerns across ML, backend, frontend, and browser layers. It combines strong model performance with explainability and multi-input support to deliver a practical, real-world cybersecurity tool.
