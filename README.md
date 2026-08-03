# 🛡️ PhishShield — Intelligent Phishing Detection System

> **Role:** Backend Developer

A machine learning-based cybersecurity system that detects phishing URLs, emails, and SMS messages using REST APIs, real-time risk classification, and browser integration.

## 📌 Overview

PhishShield is a modular phishing detection platform that combines machine learning with a Flask-based backend to analyze URLs and text for phishing attempts. As the **Backend Developer**, my primary responsibility was designing and implementing the backend architecture, REST APIs, database integration, and communication between the frontend, browser extension, and machine learning models.

---

# 👨‍💻 My Contributions (Backend Development)

* Designed and developed RESTful APIs using **Flask**
* Integrated machine learning models into backend services
* Implemented URL, Email, and SMS prediction endpoints
* Developed database integration for storing detection history
* Built modular service architecture for maintainability
* Implemented request validation and error handling
* Connected the Chrome extension and frontend with backend APIs
* Configured backend application structure and environment settings

---

# 🎯 Project Objectives

* Detect phishing URLs using machine learning
* Analyze Email and SMS phishing attempts
* Provide prediction confidence scores
* Integrate browser extension with backend APIs
* Store and retrieve detection history
* Maintain a scalable backend architecture

---

# 🚀 Features

* 🔍 Real-time phishing URL detection
* 📩 Email and SMS phishing analysis
* ⚡ Flask REST API backend
* 🗂️ Detection history storage
* 📊 Confidence score for every prediction
* 🧠 Explainable AI support
* 🌐 Chrome Extension integration
* 📈 Modular and scalable architecture

---

# 🧠 System Architecture

```
┌───────────────────────────────────────────────┐
│ Presentation Layer      │ Web UI / Extension  │
├───────────────────────────────────────────────┤
│ Backend Layer           │ Flask REST APIs     │
├───────────────────────────────────────────────┤
│ Machine Learning Layer  │ Prediction Models   │
├───────────────────────────────────────────────┤
│ Database Layer          │ SQLite Storage      │
└───────────────────────────────────────────────┘
```

---

# 🔄 Backend Workflow

```
Client (Web UI / Chrome Extension)
            │
            ▼
     Flask REST API
            │
            ▼
     Input Validation
            │
            ▼
Feature Extraction Service
            │
            ▼
Machine Learning Prediction
            │
            ▼
Store Result in Database
            │
            ▼
Return JSON Response
```

---

# 📂 Project Structure

```
phishshield/
│
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── routes/
│   │   ├── predict_url.py
│   │   └── predict_text.py
│   └── services/
│       ├── predictor.py
│       └── model_loader.py
│
├── database/
│   ├── db.py
│   └── schema.sql
│
├── shared/
│   └── feature_extractor.py
│
├── frontend/
├── extension/
├── ml_model/
├── requirements.txt
└── README.md
```

---

# ⚙️ Backend Modules

## Flask REST API

Responsible for:

* Receiving prediction requests
* Validating inputs
* Calling ML prediction services
* Returning JSON responses
* Logging detections
* Managing application configuration

---

## Prediction Service

The backend communicates with trained machine learning models to classify:

* URLs
* Emails
* SMS Messages

Response includes:

* Prediction
* Confidence Score
* Explanation (when available)

---

## API Endpoints

### URL Detection

**POST** `/predict`

Request

```json
{
  "url": "http://example.com"
}
```

Response

```json
{
  "result": "phishing",
  "confidence": 0.94,
  "reason": "Contains suspicious symbols"
}
```

---

### Email / SMS Detection

**POST** `/predict-text`

Request

```json
{
  "text": "Your account has been suspended"
}
```

Response

```json
{
  "result": "phishing",
  "confidence": 0.88
}
```

---

# 🗄️ Database

Detection results are stored in SQLite.

| Field       | Description           |
| ----------- | --------------------- |
| id          | Primary Key           |
| input_value | URL or text           |
| input_type  | url / text            |
| result      | safe / phishing       |
| confidence  | Prediction confidence |
| timestamp   | Detection time        |

---

# 🛡️ Error Handling

The backend handles:

* Invalid URLs
* Empty requests
* Low-confidence predictions
* Backend exceptions
* API validation errors

---

# 🛠️ Tech Stack

### Backend

* Python
* Flask
* SQLite
* REST API

### Machine Learning

* scikit-learn
* pandas
* NumPy
* SHAP

### Frontend

* HTML
* CSS
* JavaScript

### Browser Extension

* Chrome Extension (Manifest V3)

---

# ▶️ Running the Backend

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python ml_model/src/train_url_model.py
```

## Start the Flask Server

```bash
python backend/app.py
```

---

# 📌 Backend Responsibilities

* REST API Development
* Backend Architecture
* Database Integration
* Machine Learning Integration
* Request Validation
* API Response Handling
* Error Management
* Browser Extension Communication

---

# 🏆 Project Summary

PhishShield demonstrates how a Flask backend can efficiently integrate machine learning models with web applications and browser extensions to provide real-time phishing detection. My contribution focused on designing and implementing the backend infrastructure, ensuring reliable API communication, database persistence, and seamless integration between the frontend, browser extension, and machine learning components.
