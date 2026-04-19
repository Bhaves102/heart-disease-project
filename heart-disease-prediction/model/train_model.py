"""
Heart Disease Prediction - Model Training
Uses UCI Heart Disease Dataset (Cleveland)
RandomForestClassifier with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ── Dataset ──────────────────────────────────────────────────────────────────
# UCI Heart Disease (Cleveland) - 14 features
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Inline dataset (303 samples from UCI Cleveland)
CSV_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")

def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

def train():
    df = load_data()

    # Clean
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)  # binary: 0=no disease, 1=disease

    X = df.drop("target", axis=1)
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_sc, y_train)

    # Evaluate
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # Save
    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("✅ model.pkl and scaler.pkl saved.")
    return acc

if __name__ == "__main__":
    train()
