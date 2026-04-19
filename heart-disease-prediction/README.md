# 🫀 Heart Disease Prediction — CardioScan

> An end-to-end ML web application that predicts cardiovascular disease risk using clinical biomarkers. Built with Python, Flask, RandomForest, and SQLite.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?style=flat-square&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

CardioScan is a full-stack ML web application that uses a **Random Forest Classifier** trained on the **UCI Cleveland Heart Disease Dataset** to predict whether a patient is at risk of heart disease based on 13 clinical parameters.

Key features:
- **ML Pipeline** — RandomForest with StandardScaler, 70%+ accuracy
- **Flask REST API** — `/predict` endpoint returns prediction + confidence
- **SQLite Logging** — every prediction is stored with timestamp for history tracking
- **Prediction History Page** — browse all past predictions with results
- **Clean UI** — responsive dark-theme interface with animated results

---

## 🗂️ Project Structure

```
heart-disease-prediction/
│
├── app.py                   # Flask application & routes
│
├── model/
│   ├── train_model.py       # Model training script
│   ├── heart.csv            # UCI Cleveland dataset
│   ├── model.pkl            # Trained RandomForest model
│   └── scaler.pkl           # StandardScaler for feature normalization
│
├── database/
│   ├── db.py                # SQLite connection, logging, history queries
│   └── predictions.db       # Auto-created on first run
│
├── templates/
│   ├── index.html           # Main prediction UI
│   └── history.html         # Prediction history table
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/bhaveshyede321/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (already included, but to retrain)
```bash
python model/train_model.py
```

### 4. Run the Flask app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 🧠 Model Details

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | UCI Cleveland Heart Disease (303 samples) |
| Features | 13 clinical biomarkers |
| Target | Binary (0 = No Disease, 1 = Disease) |
| Accuracy | ~70–82% (depending on split) |
| Preprocessing | StandardScaler |

### Input Features

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = Male, 0 = Female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1=Yes) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (1=Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type (1/2/3) |

---

## 🔌 API Reference

### `POST /predict`

**Request body (JSON):**
```json
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
  "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
  "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease Detected",
  "confidence": 78.5,
  "risk_level": "High",
  "message": "⚠️ Please consult a cardiologist immediately."
}
```

### `GET /api/stats`
Returns total predictions, at-risk count, and healthy count.

### `GET /api/history`
Returns last 20 predictions as JSON.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (RandomForestClassifier, StandardScaler)
- **Data:** Pandas, NumPy
- **Database:** SQLite3 (via Python's built-in `sqlite3` module)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript

---

## 📊 Dataset

UCI Machine Learning Repository — [Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease)

---

## 👤 Author

**Bhavesh Yede**
- 📧 bhaveshyede321@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/bhaveshyede)
- 🐙 [GitHub](https://github.com/bhaveshyede321)

---

> ⚠️ **Disclaimer:** This application is for educational and portfolio purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
