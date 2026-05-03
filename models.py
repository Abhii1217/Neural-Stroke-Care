from datetime import datetime, time

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import validates


db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    user_type = db.Column(db.String(20), nullable=False, index=True)
    specialization = db.Column(db.String(120), nullable=True)
    is_available = db.Column(db.Boolean, nullable=False, default=False)
    available_from = db.Column(db.Time, nullable=True)
    available_to = db.Column(db.Time, nullable=True)

    patient_records = db.relationship(
        "PatientRecord",
        backref="patient",
        lazy=True,
        cascade="all, delete-orphan",
        foreign_keys="PatientRecord.patient_id",
    )

    @validates("user_type")
    def validate_user_type(self, key, value):
        allowed = {"patient", "doctor"}
        normalized = (value or "").strip().lower()
        if normalized not in allowed:
            raise ValueError("user_type must be either 'patient' or 'doctor'")
        return normalized

    @validates("specialization")
    def validate_specialization(self, key, value):
        if self.user_type == "doctor":
            return (value or "").strip() or "General Physician"
        return None

    @validates("available_from", "available_to")
    def validate_time_fields(self, key, value):
        if value is None or isinstance(value, time):
            return value
        raise ValueError(f"{key} must be a valid time value")


class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    prediction_result = db.Column(db.String(20), nullable=False, index=True)
    risk_probability = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    hypertension = db.Column(db.Integer)
    heart_disease = db.Column(db.Integer)
    ever_married = db.Column(db.String(10))
    work_type = db.Column(db.String(20))
    residence_type = db.Column(db.String(20))
    avg_glucose_level = db.Column(db.Float)
    bmi = db.Column(db.Float)
    smoking_status = db.Column(db.String(20))
