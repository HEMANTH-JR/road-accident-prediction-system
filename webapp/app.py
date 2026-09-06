"""
Road Accident Prediction System - Flask Dashboard Backend
Serves the GIS heatmap dashboard and a live severity-prediction endpoint
backed by the trained RandomForest model.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "static", "data")

app = Flask(__name__)

# ---- Load model artifacts once at startup ----
model = joblib.load(os.path.join(MODEL_DIR, "severity_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "severity_scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "severity_encoders.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_DIR, "severity_target_encoder.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "severity_feature_cols.pkl"))

CATEGORICAL_FEATURES = [
    "Urban/Rural", "Road Type", "Weather Conditions", "Driver Age Group",
    "Driver Gender", "Vehicle Condition", "Road Condition", "Accident Cause",
    "Season", "Time of Day", "Day of Week",
]

# Sensible defaults for fields the quick-predict form doesn't ask about
DEFAULTS = {
    "Urban/Rural": "Urban",
    "Driver Age Group": "26-40",
    "Driver Gender": "Male",
    "Vehicle Condition": "Moderate",
    "Accident Cause": "Speeding",
    "Season": "Winter",
    "Day of Week": "Monday",
    "Number of Vehicles Involved": 2,
    "Pedestrians Involved": 0,
    "Cyclists Involved": 0,
    "Traffic Volume": 500,
    "Population Density": 3000,
}

MONTH_TO_SEASON_HOUR = {
    "Morning": (8, "Spring"), "Afternoon": (14, "Spring"),
    "Evening": (19, "Spring"), "Night": (1, "Spring"),
}


def _load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/kpis")
def api_kpis():
    return jsonify(_load_json("kpis.json"))


@app.route("/api/heatmap")
def api_heatmap():
    return jsonify(_load_json("heatmap_points.json"))


@app.route("/api/blackspots")
def api_blackspots():
    return jsonify(_load_json("blackspots.json"))


@app.route("/api/time-risk")
def api_time_risk():
    return jsonify(_load_json("time_risk.json"))


@app.route("/api/country-centers")
def api_country_centers():
    return jsonify(_load_json("country_centers.json"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)

    row = dict(DEFAULTS)
    row.update({
        "Road Type": payload.get("road_type", "Highway"),
        "Weather Conditions": payload.get("weather", "Clear"),
        "Road Condition": payload.get("road_condition", "Dry"),
        "Time of Day": payload.get("time_of_day", "Afternoon"),
        "Speed Limit": float(payload.get("speed_limit", 60)),
        "Driver Alcohol Level": float(payload.get("alcohol_level", 0.0)),
        "Driver Fatigue": int(payload.get("fatigue", 0)),
        "Visibility Level": float(payload.get("visibility", 8)),
    })

    hour, season = MONTH_TO_SEASON_HOUR.get(row["Time of Day"], (14, "Spring"))
    row["Season"] = season

    month_num = 6  # mid-year placeholder for single-point prediction
    row["Month_sin"] = np.sin(2 * np.pi * month_num / 12)
    row["Month_cos"] = np.cos(2 * np.pi * month_num / 12)
    row["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["Is_Weekend"] = 1 if row["Day of Week"] in ("Saturday", "Sunday") else 0
    row["Is_Night"] = 1 if row["Time of Day"] == "Night" else 0
    row["People_Involved"] = (
        row["Number of Vehicles Involved"] + row["Pedestrians Involved"] + row["Cyclists Involved"]
    )
    row["High_Alcohol"] = 1 if row["Driver Alcohol Level"] > 0.08 else 0
    row["Poor_Visibility"] = 1 if row["Visibility Level"] < 5 else 0
    row["Adverse_Weather"] = 1 if row["Weather Conditions"] in ("Rainy", "Snowy", "Foggy", "Windy") else 0
    row["Bad_Road_Condition"] = 1 if row["Road Condition"] in ("Wet", "Icy", "Snow-covered") else 0

    df_row = pd.DataFrame([row])
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        val = str(df_row.at[0, col])
        if val not in le.classes_:
            val = le.classes_[0]  # fallback for unseen category
        df_row[col] = le.transform([val])

    X = df_row[feature_cols]
    X_scaled = scaler.transform(X)

    pred_idx = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    severity = target_encoder.inverse_transform([pred_idx])[0]

    probs = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(target_encoder.classes_, proba)
    }

    return jsonify({
        "severity": severity,
        "probabilities": probs,
        "confidence": round(float(max(proba)) * 100, 1),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
