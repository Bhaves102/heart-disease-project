"""
app.py — Heart Disease Prediction Web App
Flask + RandomForest + SQLite
Author: Bhavesh Yede
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# ── Load model & scaler ───────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ── Database ──────────────────────────────────────────────────────────────────
from database.db import init_db, log_prediction, get_history, get_stats
init_db()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


@app.route("/")
def index():
    stats = get_stats()
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features in correct order
        features = {k: float(data[k]) for k in FEATURE_ORDER}
        import pandas as pd
        input_df    = pd.DataFrame([features], columns=FEATURE_ORDER)

        # Scale + predict
        input_scaled = scaler.transform(input_df)
        prediction  = int(model.predict(input_scaled)[0])
        confidence  = round(float(model.predict_proba(input_scaled)[0][prediction]) * 100, 2)

        # Log to DB
        log_prediction(features, prediction, confidence)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "risk_level": "High" if confidence >= 75 else "Moderate" if confidence >= 55 else "Low",
            "message": (
                "⚠️ Please consult a cardiologist immediately."
                if prediction == 1
                else "✅ Your heart health indicators look normal."
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history")
def history():
    rows = get_history(limit=20)
    return render_template("history.html", rows=rows)


@app.route("/api/history")
def api_history():
    return jsonify(get_history(limit=20))


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
