# 🧠 Neural Stroke Care

## 📌 Overview

Neural Stroke Care is a full-stack machine learning web application designed to predict stroke risk using demographic and medical attributes.

The system integrates:

- A trained Logistic Regression model
- SMOTE for class imbalance correction
- Flask backend
- Authentication system for doctors and patients
- SQLite database
- Render deployment configuration

🔗 **Live Application:**  
👉 https://neural-stroke-care.onrender.com

---

## 🏗 Architecture

User → Flask Backend → Preprocessing Pipeline → SMOTE → Logistic Regression → Prediction → Result Dashboard

---

## 🧠 Machine Learning Model

### Algorithm Used
- **Logistic Regression**
- Solver: `saga`
- Max iterations: 2000
- Class weight: `balanced`
- Regularization strength: `C=1.0`

### Imbalance Handling
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- Sampling strategy: `0.6`
- Random state: `42`

### Preprocessing
- OneHot Encoding for categorical features
- Numeric features passed through directly
- ColumnTransformer used for pipeline structure

### Model Performance (as defined in training script)
- Recall at 0.40 threshold: ~95%
- AUC: ~0.98

The final model is saved as a serialized pipeline using `joblib`.

---

## 📊 Features Used for Prediction

Categorical:
- Gender
- Ever Married
- Work Type
- Residence Type
- Smoking Status

Numeric:
- Age
- Hypertension
- Heart Disease
- Average Glucose Level
- BMI

---

## 🔐 Authentication System

The system supports:

- Patient signup/login
- Doctor signup/login
- Separate dashboards
- Test history tracking

Database:
- SQLite (`instance/users.db`)

---

## 📂 Project Structure

```
Neural_Stroke_Care/
│── app.py
│── train.py
│── evaluate.py
│── model.joblib
│── train.csv
│── test.csv
│── instance/
│   └── users.db
│── static/
│   ├── css/
│   └── js/
│── templates/
│   ├── patient_login.html
│   ├── doctor_login.html
│   ├── dashboard files
│   └── result.html
│── requirements.txt
│── Procfile
│── render.yaml
```

---

## ⚙ Tech Stack

Backend:
- Python
- Flask
- scikit-learn
- imbalanced-learn
- Pandas
- Joblib

Frontend:
- HTML
- CSS
- JavaScript

Database:
- SQLite

Deployment:
- Render

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Neural_Stroke_Care.git
cd Neural_Stroke_Care
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
```

Activate:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train Model (Optional)
```bash
python train.py
```

### 5️⃣ Run Application
```bash
python app.py
```

Open:
```
http://127.0.0.1:5000
```

---

## 🌐 Deployment

Configured for deployment using:

- `Procfile`
- `render.yaml`

Live App:
👉 https://neural-stroke-care.onrender.com

---

## ⚠ Disclaimer

This application is intended for educational and research purposes only.  
It does not substitute professional medical diagnosis or treatment.

---

## 👤 Author

Your Name  
GitHub: https://github.com/Abhii1217