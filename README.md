<div align="center">

# Neural Stroke Care

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/SMOTE-Imbalanced--Learn-7952B3?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

### 🚀 [**Live Demo → neural-stroke-caree.onrender.com**](https://neural-stroke-caree.onrender.com/)

<br/>

> **Neural Stroke Care** is an AI-powered web application that leverages machine learning to assess a patient's stroke risk based on clinical and lifestyle parameters. It bridges patients and medical professionals through real-time risk prediction, doctor availability tracking, and intelligent hospital location services.

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ System Architecture & Flow](#️-system-architecture--flow)
- [🗄️ Database Schema](#️-database-schema)
- [🤖 ML Model Pipeline](#-ml-model-pipeline)
- [🚀 Getting Started](#-getting-started)
- [🌐 API Endpoints](#-api-endpoints)
- [📊 Model Evaluation](#-model-evaluation)
- [👥 Collaborators](#-collaborators)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Stroke Risk Prediction** | ML model (Logistic Regression + SMOTE) predicts stroke likelihood with probability score |
| 🔐 **Dual Auth System** | Separate signup/login flows for Patients and Doctors |
| 📋 **Patient Dashboard** | View latest prediction, history, and browse available doctors |
| 🩺 **Doctor Dashboard** | Monitor high-risk patients, toggle availability, manage schedule |
| 🏥 **Nearby Hospital Finder** | Real-time geolocation-based hospital search via OpenStreetMap Overpass API |
| 📜 **Test History** | Complete audit trail of all stroke assessments per patient |
| 🔍 **Doctor Directory** | Filterable doctor list by specialization and availability |
| 📈 **Risk Gauge** | Visual probability indicator displayed on result page |

---

## 🗂️ Project Structure

```
Neural Stroke Care/
│
├── app.py                          # 🚀 Main Flask application — routes & business logic
├── models.py                       # 🗃️  SQLAlchemy ORM models (User, PatientRecord)
├── train.py                        # 🤖 ML training script (SMOTE + Logistic Regression)
├── evaluate.py                     # 📊 Basic model evaluation
├── evaluate_advanced.py            # 📈 Advanced evaluation (ROC, PR curves, confusion matrix)
├── model.joblib                    # 💾 Serialized trained ML pipeline
├── requirements.txt                # 📦 Python dependencies
│
├── instance/
│   └── users.db                    # 🗄️  SQLite database (auto-created)
│
├── static/
│   ├── css/
│   │   └── app.css                 # 🎨 Application stylesheets
│   └── js/
│       └── app.js                  # ⚡ Frontend JavaScript (hospital finder, UI logic)
│
├── templates/                      # 🖼️  Jinja2 HTML Templates
│   ├── base.html                   # Layout base template (navbar, flash messages)
│   ├── landing.html                # Public landing page
│   ├── index.html                  # Stroke prediction form (patients)
│   ├── result.html                 # Prediction result & risk gauge
│   ├── patient_signup.html         # Patient registration
│   ├── patient_login.html          # Patient login
│   ├── patient_dashboard.html      # Patient home dashboard
│   ├── doctor_signup.html          # Doctor registration
│   ├── doctor_login.html           # Doctor login
│   ├── doctor_dashboard.html       # Doctor home dashboard
│   ├── doctors.html                # Doctor directory (filterable)
│   ├── patients.html               # All patient records (doctor view)
│   └── test_history.html           # Patient test history
│
├── Model Evaluation Images/        # 📉 Evaluation artifacts
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── Stroke_Model_Evaluation_Report.pdf
│
├── train.csv                       # 📂 Training dataset
├── test.csv                        # 📂 Test dataset
├── sample_submission.csv           # 📂 Sample submission file
└── Stroke Prediction Using Python.ipynb  # 📓 Exploratory Jupyter Notebook
```

---

## ⚙️ System Architecture & Flow

### 🔄 Full Application Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Browser)                            │
│                                                                         │
│   landing.html ──► [Patient / Doctor] ──► signup / login                │
│                                                                         │
│   Patient Flow:                      Doctor Flow:                       │
│   index.html (form) ──────────────   doctor_dashboard.html              │
│        │                             patients.html                      │
│        ▼                             doctors.html                       │
│   POST /predict                      toggle-availability                │
│        │                                                                │
│   result.html (risk gauge)                                              │
│   test_history.html                                                     │
│   doctors.html                                                          │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │  HTTP Requests (Jinja2 / JSON)
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BACKEND (Flask — app.py)                         │
│                                                                         │
│  Route Handlers:                                                        │
│  ┌────────────────────────────────────────────────┐                     │
│  │  /              → landing / index              │                     │
│  │  /signup/<type> → patient or doctor signup     │                     │
│  │  /login/<type>  → session-based authentication │                     │
│  │  /logout        → session.clear()              │                     │
│  │  /predict       → ML inference + DB write      │                     │
│  │  /dashboard     → role-based dashboard         │                     │
│  │  /doctors       → doctor directory + filters   │                     │
│  │  /patients      → all records (doctor only)    │                     │
│  │  /test_history  → patient record history       │                     │
│  │  /hospitals     → Overpass API proxy (JSON)    │                     │
│  │  /doctor/toggle-availability → update doctor   │                     │
│  └────────────────────────────────────────────────┘                     │
│                                                                         │
│  Middleware & Helpers:                                                  │
│  • current_user()     → resolves user from session                      │
│  • require_login()    → auth guard for protected routes                 │
│  • predict_stroke_risk() → calls ML pipeline                            │
│  • ensure_schema()    → safe DB migrations                              │
└────────────┬──────────────────────────┬─────────────────────────────────┘
             │                          │
             ▼                          ▼
┌────────────────────┐     ┌────────────────────────────────────────────┐
│   ML Engine        │     │       DATABASE (SQLite — users.db)         │
│                    │     │                                            │
│  model.joblib      │     │  ORM via Flask-SQLAlchemy (models.py)      │
│  ┌──────────────┐  │     │                                            │
│  │ Preprocessor │  │     │  Tables:                                   │
│  │ (OneHotEnc.) │  │     │  • user           (patients & doctors)     │
│  │ SMOTE        │  │     │  • patient_record (predictions & inputs)   │
│  │ LogisticReg  │  │     │                                            │
│  └──────────────┘  │     └────────────────────────────────────────────┘
│                    │
│  Output:           │     ┌────────────────────────────────────────────┐
│  • "Likely" /      │     │     EXTERNAL API                           │
│    "Not Likely"    │     │                                            │
│  • Risk % score    │     │  OpenStreetMap Overpass API                │
└────────────────────┘     │  → Nearby hospital geolocation search      │
                           │  → Haversine distance calculation          │
                           │  → Returns top 10 hospitals within 8km     │
                           └────────────────────────────────────────────┘
```

### 🔐 Authentication Flow

```
User visits /signup/<type>
        │
        ▼
Fill form ──► POST /signup/<type>
        │
        ├──► Email exists? ──► Flash error → redirect
        │
        └──► Hash password (Werkzeug)
             Save User to DB
             Flash success → redirect to /login/<type>
                                   │
                              POST /login/<type>
                                   │
                              Match email + password hash?
                                   │
                    ┌──────────────┴──────────────┐
                   YES                            NO
                    │                              │
              session["user_id"] = id        Flash "Invalid credentials"
              session["user_type"] = type         │
                    │                        Redirect to login
              redirect /dashboard
                    │
           ┌────────┴────────┐
      user_type=patient   user_type=doctor
           │                    │
   patient_dashboard.html  doctor_dashboard.html
```

---

## 🗄️ Database Schema

### Table: `user`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PK, Auto-increment | Unique user identifier |
| `name` | VARCHAR(100) | NOT NULL | Full name |
| `email` | VARCHAR(120) | UNIQUE, NOT NULL, Indexed | Login email |
| `password` | VARCHAR(200) | NOT NULL | Werkzeug-hashed password |
| `user_type` | VARCHAR(20) | NOT NULL, Indexed | `"patient"` or `"doctor"` |
| `specialization` | VARCHAR(120) | Nullable | Doctor specialization (e.g., Neurology) |
| `is_available` | BOOLEAN | NOT NULL, Default: False | Doctor online/offline status |
| `available_from` | TIME | Nullable | Doctor availability start time |
| `available_to` | TIME | Nullable | Doctor availability end time |

### Table: `patient_record`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PK, Auto-increment | Record identifier |
| `patient_id` | INTEGER | FK → user.id, Indexed | Associated patient |
| `prediction_result` | VARCHAR(20) | NOT NULL, Indexed | `"Likely"` or `"Not Likely"` |
| `risk_probability` | FLOAT | Nullable | Stroke risk percentage (0–100) |
| `created_at` | DATETIME | NOT NULL, Indexed | UTC timestamp of assessment |
| `gender` | VARCHAR(10) | — | Patient gender |
| `age` | INTEGER | — | Patient age |
| `hypertension` | INTEGER | — | 0 or 1 |
| `heart_disease` | INTEGER | — | 0 or 1 |
| `ever_married` | VARCHAR(10) | — | `"yes"` or `"no"` |
| `work_type` | VARCHAR(20) | — | Employment category |
| `residence_type` | VARCHAR(20) | — | `"urban"` or `"rural"` |
| `avg_glucose_level` | FLOAT | — | Average blood glucose (mg/dL) |
| `bmi` | FLOAT | — | Body Mass Index |
| `smoking_status` | VARCHAR(20) | — | Smoking history category |

### Entity Relationship Diagram

```
┌─────────────────────────────────┐         ┌─────────────────────────────────────┐
│             USER                │         │          PATIENT_RECORD             │
│─────────────────────────────────│         │─────────────────────────────────────│
│ PK  id              INTEGER     │◄──┐     │ PK  id                  INTEGER     │
│     name            VARCHAR(100)│   └─────│ FK  patient_id          INTEGER     │
│     email           VARCHAR(120)│         │     prediction_result   VARCHAR(20) │
│     password        VARCHAR(200)│         │     risk_probability    FLOAT       │
│     user_type       VARCHAR(20) │         │     created_at          DATETIME    │
│     specialization  VARCHAR(120)│         │     gender              VARCHAR(10) │
│     is_available    BOOLEAN     │         │     age                 INTEGER     │
│     available_from  TIME        │         │     hypertension        INTEGER     │
│     available_to    TIME        │         │     heart_disease       INTEGER     │
└─────────────────────────────────┘         │     ever_married        VARCHAR(10) │
                                            │     work_type           VARCHAR(20) │
              1 ──────────── many           │     residence_type      VARCHAR(20) │
                                            │     avg_glucose_level   FLOAT       │
                                            │     bmi                 FLOAT       │
                                            │     smoking_status      VARCHAR(20) │
                                            └─────────────────────────────────────┘
```

---

## 🤖 ML Model Pipeline

The prediction engine is built with `scikit-learn` and `imbalanced-learn`, serialized as `model.joblib`.

```
Input Features (10)
    │
    ├── Categorical: gender, ever_married, work_type, Residence_type, smoking_status
    └── Numerical:  age, hypertension, heart_disease, avg_glucose_level, bmi
         │
         ▼
┌────────────────────────────────────────┐
│  Step 1: ColumnTransformer             │
│  • OneHotEncoder  → categorical cols   │
│  • Passthrough    → numerical cols     │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  Step 2: SMOTE                         │
│  • sampling_strategy = 0.6             │
│  • random_state = 42                   │
│  • Handles class imbalance             │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  Step 3: Logistic Regression           │
│  • solver = "saga"                     │
│  • C = 1.0                             │
│  • class_weight = "balanced"           │
│  • max_iter = 2000                     │
└────────────────────┬───────────────────┘
                     │
                     ▼
         predict_proba(input)[0][1]
                     │
           ┌─────────┴─────────┐
        prob ≥ 0.40         prob < 0.40
           │                    │
       "Likely"           "Not Likely"
     (High Risk)           (Low Risk)
```

**Key Metrics:**
- 🎯 Recall @ 0.40 threshold: ~**95%**
- 📈 AUC: ~**0.98**
- ⚖️ Class imbalance handled via SMOTE + balanced class weights

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/neural-stroke-care.git
cd neural-stroke-care

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain the model
python train.py

# 5. Run the application
python app.py
```

The app will be available at `http://127.0.0.1:5000`

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `change-this-secret-key-123` | Flask session secret — **change in production** |
| `SQLALCHEMY_DATABASE_URI` | `sqlite:///users.db` | Database connection string |

---

## 🌐 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | Public | Landing page or prediction form |
| `GET/POST` | `/signup/<type>` | Public | Register as `patient` or `doctor` |
| `GET/POST` | `/login/<type>` | Public | Login as `patient` or `doctor` |
| `GET` | `/logout` | Session | Clear session and redirect |
| `POST` | `/predict` | Patient only | Submit stroke risk assessment |
| `GET` | `/dashboard` | Session | Role-based dashboard |
| `GET` | `/doctors` | Session | Browse doctor directory |
| `GET` | `/patients` | Doctor only | View all patient records |
| `GET` | `/test_history` | Session | Patient's prediction history |
| `POST` | `/doctor/toggle-availability` | Doctor only | Update availability status |
| `GET` | `/hospitals?lat=&lon=` | Public | Nearby hospital JSON API |

---

## 📊 Model Evaluation

The `Model Evaluation Images/` folder contains full evaluation artifacts:

| Artifact | Description |
|---|---|
| `confusion_matrix.png` | True/False Positive & Negative breakdown |
| `roc_curve.png` | Receiver Operating Characteristic curve (AUC ~0.98) |
| `pr_curve.png` | Precision-Recall curve |
| `Stroke_Model_Evaluation_Report.pdf` | Comprehensive evaluation report |

Run evaluations manually:

```bash
python evaluate.py           # Basic metrics
python evaluate_advanced.py  # ROC, PR curves, confusion matrix
```

---

## 👥 Collaborators

<div align="center">

| 👤 Name | 🎓 Role |
|---|---|
| **Abhishek Anand** | Team Member |
| **Bibek Das** | Team Member |
| **Debarghya Datta** | Team Member |
| **Kumari Snehlata** | Team Member |
| **Aditya Kumar Singh** | Team Member |
| **Aditya Bhardwaj** | Team Member |

</div>

---

<div align="center">

Made with ❤️ and 🧠 by the **Neural Stroke Care Team**

*Empowering early stroke detection through the power of AI*

</div>
