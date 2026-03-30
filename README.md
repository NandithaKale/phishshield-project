🛡️ PhishShield - Intelligent Phishing Detection System

📌 Introduction



PhishShield is a machine learning-based cybersecurity system designed to detect phishing attacks in URLs. The system analyzes structural and lexical features of URLs and predicts whether they are Safe, Moderate Risk, or Phishing.



The project follows a modular and scalable architecture and is designed to be extended into a full real-world application including backend APIs, browser extensions, and multi-platform phishing detection.



🎯 Objectives

Detect phishing URLs using machine learning

Provide risk-based classification with confidence score

Build a scalable system architecture (ML + Backend + Frontend)

Extend detection to browser extension and real-time usage

Provide explainability for predictions (future scope)

🧠 System Architecture



PhishShield follows a layered architecture similar to real-world systems:



🔹 Layers

Presentation Layer

Web interface / Browser extension

Accepts user input (URL)

API Layer (Backend)

Handles requests using Flask

Communicates with ML model

Machine Learning Layer

Feature extraction

Model prediction (Random Forest)

Risk scoring

Data Layer

Stores URL history and results (future scope)

🔄 System Workflow

User enters or visits a URL

Request is sent to backend

URL features are extracted

ML model predicts probability

Risk classification is applied

Result is returned with confidence

(Optional) Data stored for history

📂 Project Structure



phishshield\_project/



│

├── ml\_model/

│ ├── dataset/

│ │ └── phishing\_dataset.csv

│ │

│ ├── src/

│ │ ├── train\_url\_model.py

│ │ ├── predict\_url.py

│ │ └── model\_utils.py

│ │

│ └── saved\_model/

│ └── url\_model.pkl



│

├── backend/

│ ├── app.py

│ ├── routes/

│ │ ├── url\_routes.py

│ │ ├── email\_routes.py

│ │ └── sms\_routes.py

│ │

│ ├── services/

│ │ ├── prediction\_service.py

│ │ ├── explainability\_service.py

│ │ └── history\_service.py

│ │

│ └── database/

│ └── db\_config.py



│

├── frontend/

│ ├── templates/

│ │ ├── index.html

│ │ ├── result.html

│ │ └── dashboard.html

│ │

│ └── static/

│ ├── css/

│ └── js/



│

├── extension/

│ ├── manifest.json

│ ├── popup.html

│ ├── popup.js

│ └── background.js



│

├── shared/

│ ├── feature\_extractor.py

│ ├── utils.py

│ └── constants.py



│

├── requirements.txt

└── README.md



⚙️ Core Modules

🔹 Feature Extraction Module



Extracts important URL features such as:



URL length

Number of dots, digits, special characters

Presence of HTTPS

Suspicious keywords (login, verify, bank, etc.)

Suspicious domain extensions (.xyz, .tk, .ml, etc.)

🔹 Machine Learning Module

Algorithm: Random Forest Classifier

Input: Extracted URL features

Output: Probability of phishing

🔹 Risk Classification Module

0–30% → Safe

30–70% → Moderate Risk

70–100% → Phishing

🔹 Backend API Module (Planned)

Flask-based API

Handles prediction requests

Connects frontend with ML model

🔹 Browser Extension Module (Planned)

Detects current website URL

Sends request to backend

Displays real-time phishing warning

🔹 Explainability Module (Future)

Explains why a URL is flagged

Highlights suspicious features

🤖 Machine Learning Workflow

Training Phase

Load dataset

Extract features

Train Random Forest model

Save trained model (.pkl)

Prediction Phase

Input URL

Extract features

Load trained model

Predict probability

Convert to risk level

📊 Model Performance

Accuracy : \~84% (training dataset)

Accuracy (test cases) : \~96.67%

Precision: 1.0

Recall : 0.93

F1 Score : 0.96

🚀 Features

✅ Implemented

URL phishing detection

Feature engineering

ML model training and prediction

Confidence score output

Risk-based classification

🔄 In Progress

Flask backend API

Frontend UI integration

🔮 Future Enhancements

Browser extension

Email/SMS phishing detection

Explainable AI

Dashboard for analytics

Database integration

🛠️ Tech Stack

Python

scikit-learn

pandas, numpy

Flask (planned)

HTML, CSS, JavaScript (planned)

▶️ How to Run

Install dependencies

pip install -r requirements.txt

Train model

python ml\_model/src/train\_url\_model.py

👥 Team Roles

ML Developer: Feature engineering, model training, evaluation

Backend Developer: API development and integration

Frontend Developer: UI and user interaction

🏆 Conclusion



PhishShield is a scalable and intelligent phishing detection system that leverages machine learning to identify malicious URLs. The current system demonstrates strong performance and provides a foundation for building a complete cybersecurity solution with real-world applications.

