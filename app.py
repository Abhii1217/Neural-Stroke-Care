from datetime import datetime

import joblib
import pandas as pd
import requests
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from math import atan2, cos, radians, sin, sqrt
from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash

from models import PatientRecord, User, db

import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key-123"
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///users.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

model_data = joblib.load("model.joblib")
pipeline = model_data["model"]


def parse_time_or_none(value):
    value = (value or "").strip()
    if not value:
        return None
    return datetime.strptime(value, "%H:%M").time()


def current_user():
    if "user_id" not in session:
        return None
    return db.session.get(User, session["user_id"])


def require_login():
    user = current_user()
    if not user:
        return None, redirect(url_for("index"))
    return user, None


def predict_stroke_risk(data):
    df = pd.DataFrame([data])
    prob = pipeline.predict_proba(df)[0][1]
    return ("Likely" if prob >= 0.40 else "Not Likely"), round(prob * 100, 2)


def ensure_schema():
    inspector = inspect(db.engine)
    if not inspector.has_table("user"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("user")}
    statements = []
    if "specialization" not in existing_columns:
        statements.append("ALTER TABLE user ADD COLUMN specialization VARCHAR(120)")
    if "is_available" not in existing_columns:
        statements.append("ALTER TABLE user ADD COLUMN is_available BOOLEAN DEFAULT 0 NOT NULL")
    if "available_from" not in existing_columns:
        statements.append("ALTER TABLE user ADD COLUMN available_from TIME")
    if "available_to" not in existing_columns:
        statements.append("ALTER TABLE user ADD COLUMN available_to TIME")

    for statement in statements:
        db.session.execute(text(statement))
    if statements:
        db.session.commit()


@app.route("/", methods=["GET"])
def index():
    if "user_id" not in session:
        return render_template("landing.html")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user, redirect_response = require_login()
    if redirect_response:
        return redirect_response
    if user.user_type != "patient":
        flash("Only patients can submit stroke assessments.")
        return redirect(url_for("dashboard"))

    try:
        data = {
            "gender": request.form["gender"].lower(),
            "age": int(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "ever_married": request.form["ever_married"].lower(),
            "work_type": request.form["work_type"],
            "Residence_type": request.form["residence_type"].lower(),
            "avg_glucose_level": float(request.form["avg_glucose_level"]),
            "bmi": float(request.form["bmi"]),
            "smoking_status": request.form["smoking_status"].lower(),
        }
    except (KeyError, ValueError):
        flash("Please fill all prediction fields correctly.")
        return redirect(url_for("index"))

    work_map = {
        "Government job": "Govt_job",
        "Never Worked": "Never_worked",
        "Self-employed": "Self-employed",
        "Children": "children",
        "Private": "Private",
    }
    data["work_type"] = work_map.get(data["work_type"], data["work_type"])
    result, risk = predict_stroke_risk(data)

    save_data = data.copy()
    save_data["residence_type"] = save_data.pop("Residence_type")

    new_record = PatientRecord(
        patient_id=user.id,
        prediction_result=result,
        risk_probability=risk,
        **save_data,
    )
    db.session.add(new_record)
    db.session.commit()

    return render_template(
        "result.html",
        result=result,
        probability=risk,
        input_data=save_data,
        date=datetime.utcnow().strftime("%B %d, %Y"),
    )


@app.route("/signup/<user_type>", methods=["GET", "POST"])
def signup(user_type):
    user_type = (user_type or "").lower()
    if user_type not in {"patient", "doctor"}:
        flash("Invalid account type.")
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form["email"].strip().lower()
        if User.query.filter_by(email=email).first():
            flash("Email already exists")
            return redirect(url_for("signup", user_type=user_type))

        try:
            user = User(
                email=email,
                password=generate_password_hash(request.form["password"]),
                user_type=user_type,
                name=request.form["name"].strip(),
                specialization=request.form.get("specialization"),
                is_available=bool(request.form.get("is_available")) if user_type == "doctor" else False,
                available_from=parse_time_or_none(request.form.get("available_from")),
                available_to=parse_time_or_none(request.form.get("available_to")),
            )
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("signup", user_type=user_type))

        db.session.add(user)
        db.session.commit()
        flash("Account created successfully!")
        return redirect(url_for("login", user_type=user_type))

    return render_template(f"{user_type}_signup.html")


@app.route("/login/<user_type>", methods=["GET", "POST"])
def login(user_type):
    user_type = (user_type or "").lower()
    if user_type not in {"patient", "doctor"}:
        flash("Invalid login type.")
        return redirect(url_for("index"))

    if request.method == "POST":
        user = User.query.filter_by(email=request.form["email"].strip().lower(), user_type=user_type).first()
        if user and check_password_hash(user.password, request.form["password"]):
            session["user_id"] = user.id
            session["user_type"] = user.user_type
            return redirect(url_for("dashboard"))
        flash("Invalid email or password")
    return render_template(f"{user_type}_login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    user, redirect_response = require_login()
    if redirect_response:
        return redirect_response

    if user.user_type == "doctor":
        likely_patient_rows = (
            db.session.query(PatientRecord, User)
            .join(User, User.id == PatientRecord.patient_id)
            .filter(User.user_type == "patient", PatientRecord.prediction_result == "Likely")
            .order_by(PatientRecord.created_at.desc())
            .all()
        )
        return render_template("doctor_dashboard.html", user=user, likely_patient_rows=likely_patient_rows)

    latest_record = (
        PatientRecord.query.filter_by(patient_id=user.id)
        .order_by(PatientRecord.created_at.desc())
        .first()
    )
    history = (
        PatientRecord.query.filter_by(patient_id=user.id)
        .order_by(PatientRecord.created_at.desc())
        .limit(5)
        .all()
    )
    doctors = User.query.filter_by(user_type="doctor").order_by(User.is_available.desc(), User.name.asc()).all()
    return render_template(
        "patient_dashboard.html",
        user=user,
        latest_test=latest_record,
        history=history,
        doctors=doctors,
    )


@app.route("/doctors")
def doctors():
    user, redirect_response = require_login()
    if redirect_response:
        return redirect_response

    availability = (request.args.get("availability") or "all").lower()
    specialization = (request.args.get("specialization") or "").strip()

    query = User.query.filter_by(user_type="doctor")
    if availability == "online":
        query = query.filter_by(is_available=True)
    elif availability == "offline":
        query = query.filter_by(is_available=False)
    if specialization:
        query = query.filter(User.specialization.ilike(f"%{specialization}%"))

    doctors_list = query.order_by(User.is_available.desc(), User.name.asc()).all()
    specializations = [
        row[0]
        for row in db.session.query(User.specialization)
        .filter(User.user_type == "doctor", User.specialization.isnot(None))
        .distinct()
        .order_by(User.specialization.asc())
        .all()
        if row[0]
    ]
    return render_template(
        "doctors.html",
        doctors=doctors_list,
        selected_availability=availability,
        selected_specialization=specialization,
        specializations=specializations,
    )


@app.route("/doctor/toggle-availability", methods=["POST"])
def toggle_availability():
    user, redirect_response = require_login()
    if redirect_response:
        return redirect_response
    if user.user_type != "doctor":
        flash("Only doctors can update availability.")
        return redirect(url_for("dashboard"))

    user.is_available = not user.is_available
    user.specialization = (request.form.get("specialization") or user.specialization or "General Physician").strip()
    user.available_from = parse_time_or_none(request.form.get("available_from")) or user.available_from
    user.available_to = parse_time_or_none(request.form.get("available_to")) or user.available_to
    db.session.commit()
    flash(f"Availability updated to {'Online' if user.is_available else 'Offline'}.")
    return redirect(url_for("dashboard"))


@app.route("/patients")
def patients():
    user, redirect_response = require_login()
    if redirect_response:
        return redirect_response
    if user.user_type != "doctor":
        flash("Only doctors can access patient records.")
        return redirect(url_for("dashboard"))

    patient_records = (
        db.session.query(PatientRecord, User)
        .join(User, User.id == PatientRecord.patient_id)
        .filter(User.user_type == "patient")
        .order_by(PatientRecord.created_at.desc())
        .all()
    )
    return render_template("patients.html", patient_records=patient_records, user=user)


@app.route("/test_history")
def test_history():
    user, redirect_response = require_login()
    if redirect_response:
        return redirect_response
    tests = (
        PatientRecord.query.filter_by(patient_id=user.id)
        .order_by(PatientRecord.created_at.desc())
        .all()
    )
    return render_template("test_history.html", user=user, tests=tests)


@app.route("/hospitals")
def hospitals():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify([])

    query = (
        "[out:json][timeout:30];"
        "("
        f"node['amenity'='hospital'](around:8000,{lat},{lon});"
        f"way['amenity'='hospital'](around:8000,{lat},{lon});"
        f"node['healthcare'='hospital'](around:8000,{lat},{lon});"
        ");"
        "out center;"
    )

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=20,
            headers={"User-Agent": "StrokeApp/1.0"},
        )
        data = response.json()
    except Exception:
        return jsonify([])

    results = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name", "Unnamed Hospital")
        if any(word in name.lower() for word in ["eye", "dental", "clinic", "optical", "vision"]):
            continue

        lat2 = element.get("lat") or element.get("center", {}).get("lat")
        lon2 = element.get("lon") or element.get("center", {}).get("lon")
        if not (lat2 and lon2):
            continue

        earth_radius_km = 6371
        dlat = radians(lat2 - lat)
        dlon = radians(lon2 - lon)
        a = sin(dlat / 2) ** 2 + cos(radians(lat)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        distance = earth_radius_km * 2 * atan2(sqrt(a), sqrt(1 - a))

        results.append(
            {
                "name": name,
                "address": tags.get("addr:full") or tags.get("addr:street") or "Address not available",
                "distance": round(distance, 1),
            }
        )

    results.sort(key=lambda x: x["distance"])
    return jsonify(results[:10])


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        ensure_schema()
    app.run(debug=True)