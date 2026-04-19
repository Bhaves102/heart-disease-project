"""
database/db.py
Handles SQLite connection and prediction history logging.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            age         INTEGER,
            sex         INTEGER,
            cp          INTEGER,
            trestbps    INTEGER,
            chol        INTEGER,
            fbs         INTEGER,
            restecg     INTEGER,
            thalach     INTEGER,
            exang       INTEGER,
            oldpeak     REAL,
            slope       INTEGER,
            ca          INTEGER,
            thal        REAL,
            prediction  INTEGER,
            confidence  REAL,
            created_at  TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(features: dict, prediction: int, confidence: float):
    """Insert a prediction record into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            prediction, confidence, created_at
        ) VALUES (
            :age, :sex, :cp, :trestbps, :chol, :fbs, :restecg,
            :thalach, :exang, :oldpeak, :slope, :ca, :thal,
            :prediction, :confidence, :created_at
        )
    """, {**features, "prediction": prediction, "confidence": confidence,
          "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    conn.commit()
    conn.close()


def get_history(limit: int = 10):
    """Fetch recent prediction history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_stats():
    """Return summary stats for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total = cursor.fetchone()["total"]
    cursor.execute("SELECT COUNT(*) as at_risk FROM predictions WHERE prediction = 1")
    at_risk = cursor.fetchone()["at_risk"]
    conn.close()
    return {"total": total, "at_risk": at_risk, "safe": total - at_risk}
