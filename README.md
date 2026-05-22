<div align="center">

# 🧠 Neural Stroke Care

### AI-powered stroke risk prediction for patients and clinicians

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-WSGI-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://gunicorn.org)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-View%20App-e10600?style=for-the-badge&logo=render&logoColor=white)](https://neural-stroke-care.onrender.com)

</div>

---

## 📋 Overview

Neural Stroke Care is a full-stack Flask web application that combines a calibrated machine learning model with a role-based clinical interface. Patients submit health assessments and receive real-time stroke risk scores; doctors monitor high-risk patient cohorts and manage their availability. A live hospital finder powered by the OpenStreetMap Overpass API rounds out the care workflow.

The model (Logistic Regression + SMOTE, ~0.98 AUC) is trained on the Kaggle Stroke Prediction dataset and uses a 40% probability threshold optimized for high recall (~95%) over precision.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Stroke Risk Prediction** | Real-time ML inference with calibrated probability score and gauge visualization |
| **Dual Role Auth** | Separate patient and doctor signup / login flows with session-based access control |
| **Patient Dashboard** | Latest assessment, 5-test history sparkline, BMI/glucose trend cards |
| **Doctor Dashboard** | Filterable cohort view of all "Likely Stroke" patients, sorted by recency |
| **Hospital Finder** | Geolocation-based nearest hospitals via OpenStreetMap Overpass API (8 km radius) |
| **Doctor Availability** | Doctors set online/offline status, specialization, and working hours |
| **Test History** | Full chronological log of all past assessments per patient |
| **Model Evaluation** | ROC curve, PR curve, confusion matrix, and PDF report included in repo |

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Jinja2 Templates, Bootstrap 5, custom CSS (Inter font, CSS variables), Vanilla JS |
| **Backend** | Flask, Flask-SQLAlchemy, Werkzeug (password hashing), Gunicorn |
| **ML / Data** | scikit-learn, imbalanced-learn (SMOTE), pandas, NumPy, joblib |
| **Database** | SQLite (via Flask-SQLAlchemy) |
| **External APIs** | OpenStreetMap Overpass API (hospital finder) |

---

## 🏗 Architecture

```
Browser
   │
   ▼
Flask App (app.py)
   ├── Auth routes (/signup, /login, /logout)
   │       └── Werkzeug password hashing → SQLite (User table)
   │
   ├── Predict route (/predict) [POST]
   │       ├── Form input → pandas DataFrame
   │       ├── joblib pipeline (OneHotEncoder → SMOTE → LogisticRegression)
   │       │       └── Threshold: prob ≥ 0.40 → "Likely"
   │       └── PatientRecord saved → result.html rendered
   │
   ├── Dashboard (/dashboard)
   │       ├── Patient  → latest record + 5-item history + doctor list
   │       └── Doctor   → all "Likely" patient rows (joined query)
   │
   ├── Doctors & Patients (/doctors, /patients)
   │       └── Filtered queries → rendered tables
   │
   └── Hospital Finder (/hospitals) [JSON API]
           └── User lat/lon → Overpass API → Haversine sort → top 10
```

---

## 📁 Project Structure

```
Neural Stroke Care/
├── app.py                          # Flask app, all routes, prediction logic
├── models.py                       # SQLAlchemy models: User, PatientRecord
├── train.py                        # Training script (SMOTE + LogisticRegression)
├── evaluate.py                     # CLI model evaluation metrics
├── evaluate_advanced.py            # ROC/PR curves + PDF report generator
├── model.joblib                    # Serialized pipeline (preprocessor + model)
├── requirements.txt                # Python dependencies
├── Procfile                        # Gunicorn entry point for Heroku/Render
├── templates/
│   ├── base.html                   # Shared navbar, flash messages, Bootstrap
│   ├── landing.html                # Public landing page
│   ├── index.html                  # Stroke assessment form (patients only)
│   ├── result.html                 # Prediction result + risk gauge
│   ├── patient_dashboard.html      # Patient home with history + doctor list
│   ├── doctor_dashboard.html       # Doctor home with high-risk patient cohort
│   ├── doctors.html                # Filterable doctor directory
│   ├── patients.html               # Doctor-only full patient records view
│   └── test_history.html           # Paginated assessment history
├── static/
│   ├── css/app.css                 # Custom design system (CSS variables, cards)
│   └── js/app.js                   # Navbar scroll, IntersectionObserver, pw strength
├── Model Evaluation Images/
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   └── Stroke_Model_Evaluation_Report.pdf
└── Stroke Prediction Using Python.ipynb   # EDA and model experimentation notebook
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Abhii1217/Neural-Stroke-Care.git
cd Neural-Stroke-Care
pip install -r requirements.txt
```

**2. Initialize the database**
```bash
# SQLite DB is created automatically on first run — no manual step needed.
# Optionally force-create it ahead of time:
python -c "from app import app, db; app.app_context().__enter__(); db.create_all()"
```

**3. Set environment variables**

Create a `.env` file or export directly — only two variables are needed:
```bash
export SECRET_KEY="your-strong-secret-here"
export DATABASE_URL="sqlite:///users.db"   # default; can omit
```

---

## 🔐 Environment Variables

### Backend `.env`

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `change-this-secret-key-123` | Flask session secret — **must be changed in production** |
| `DATABASE_URL` | `sqlite:///users.db` | SQLAlchemy DB URI (defaults to SQLite) |

> No frontend `.env` — this is a server-rendered Jinja2 app with no separate frontend build step.

---

## 💻 Local Development

```bash
# Run the development server
python app.py
```

| Service | URL |
|---|---|
| Web App | http://localhost:5000 |
| Patient Signup | http://localhost:5000/signup/patient |
| Doctor Signup | http://localhost:5000/signup/doctor |
| Hospital API | http://localhost:5000/hospitals?lat=28.6&lon=77.2 |

---

## 📖 Usage

1. Navigate to the landing page and choose **Patient onboarding** or **Doctor workspace**.
2. Register an account; doctors additionally set their specialization and working hours.
3. As a **patient**, complete the stroke assessment form (age, BMI, glucose, lifestyle factors).
4. View your instant risk score, probability gauge, and personalized result page.
5. Revisit your full assessment history under **Test History**.
6. Use the **Hospital Finder** on the dashboard to locate nearby hospitals based on your location.
7. As a **doctor**, the dashboard surfaces all patients flagged "Likely Stroke" — filter by availability or specialization from the Doctors directory.

---

## 📜 Scripts

| Location | Command | Description |
|---|---|---|
| Root | `python app.py` | Start Flask dev server (debug=True) |
| Root | `gunicorn app:app` | Start production WSGI server |
| Root | `python train.py` | Re-train and save `model.joblib` |
| Root | `python evaluate.py` | Print accuracy, recall, AUC metrics to terminal |
| Root | `python evaluate_advanced.py` | Generate ROC/PR curves + PDF evaluation report |

---

## 🔌 API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Landing page (unauthenticated) or assessment form (authenticated) |
| `POST` | `/predict` | Submit stroke assessment; returns rendered result page |
| `GET/POST` | `/signup/<user_type>` | Register as `patient` or `doctor` |
| `GET/POST` | `/login/<user_type>` | Authenticate as `patient` or `doctor` |
| `GET` | `/logout` | Clear session and redirect |
| `GET` | `/dashboard` | Role-aware dashboard (patient or doctor) |
| `POST` | `/doctor/toggle-availability` | Toggle doctor online/offline status |
| `GET` | `/doctors` | Filterable doctor directory (availability, specialization) |
| `GET` | `/patients` | Doctor-only: all patient records |
| `GET` | `/test_history` | Patient-only: full assessment history |
| `GET` | `/hospitals?lat=&lon=` | JSON: nearest hospitals via Overpass API |

---

## 🗄 Database Setup

The schema is auto-created by SQLAlchemy on startup. Key tables:

```sql
-- User (patients and doctors share one table, distinguished by user_type)
CREATE TABLE user (
    id              INTEGER PRIMARY KEY,
    name            VARCHAR(100)  NOT NULL,
    email           VARCHAR(120)  UNIQUE NOT NULL,
    password        VARCHAR(200)  NOT NULL,
    user_type       VARCHAR(20)   NOT NULL,          -- 'patient' | 'doctor'
    specialization  VARCHAR(120),                    -- doctors only
    is_available    BOOLEAN       DEFAULT FALSE,     -- doctor availability flag
    available_from  TIME,                            -- doctor schedule start
    available_to    TIME                             -- doctor schedule end
);

-- PatientRecord (one row per assessment submission)
CREATE TABLE patient_record (
    id                  INTEGER PRIMARY KEY,
    patient_id          INTEGER REFERENCES user(id),
    prediction_result   VARCHAR(20),   -- 'Likely' | 'Not Likely'
    risk_probability    FLOAT,         -- 0.0 – 100.0
    created_at          DATETIME,
    gender              VARCHAR(10),
    age                 INTEGER,
    hypertension        INTEGER,       -- 0 | 1
    heart_disease       INTEGER,       -- 0 | 1
    ever_married        VARCHAR(10),
    work_type           VARCHAR(20),
    residence_type      VARCHAR(20),
    avg_glucose_level   FLOAT,
    bmi                 FLOAT,
    smoking_status      VARCHAR(20)
);
```

```bash
# Auto-create schema (runs automatically on app start, or manually):
python -c "from app import app, db; app.app_context().__enter__(); db.create_all()"
```

---

## ⚠️ Known Limitations

- **No email verification** — accounts are activated immediately on signup; no confirmation flow.
- **Test history capped at 5 on dashboard** — `.limit(5)` is hardcoded in the dashboard query; full history requires navigating to `/test_history`.
- **No test suite** — zero unit or integration tests in the repository.

---

## 🗺 Roadmap

- [ ] Add email verification on signup
- [ ] Replace session auth with JWT or Flask-Login for stateless API support
- [ ] Add appointment booking between patients and available doctors

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss significant changes.

```bash
# Workflow
git checkout -b feat/your-feature
git commit -m "feat: describe your change"
git push origin feat/your-feature
# Open a Pull Request against main
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for full terms.

---

## 👤 Author

Abhishek Anand & Aditya Bhardwaj

[![GitHub](https://img.shields.io/badge/GitHub-Abhii1217-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abhii1217)
[![GitHub](https://img.shields.io/badge/GitHub-Aditya--Bhardwaj--jod-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Aditya-Bhardwaj-jod)
