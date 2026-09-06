"""
Step 3: Accident Severity Classification Model.
Predicts Accident Severity (Minor / Moderate / Severe) from
road, environmental, driver, and temporal features.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA_PATH = "/home/claude/accident_project/data/accidents_features_v2.csv"
MODEL_DIR = "/home/claude/accident_project/models"

CATEGORICAL_FEATURES = [
    "Urban/Rural", "Road Type", "Weather Conditions", "Driver Age Group",
    "Driver Gender", "Vehicle Condition", "Road Condition", "Accident Cause",
    "Season", "Time of Day", "Day of Week",
]

NUMERIC_FEATURES = [
    "Visibility Level", "Number of Vehicles Involved", "Speed Limit",
    "Driver Alcohol Level", "Driver Fatigue", "Pedestrians Involved",
    "Cyclists Involved", "Traffic Volume", "Population Density",
    "Month_sin", "Month_cos", "Hour_sin", "Hour_cos",
    "Is_Weekend", "Is_Night", "People_Involved", "High_Alcohol",
    "Poor_Visibility", "Adverse_Weather", "Bad_Road_Condition",
]

TARGET = "Accident Severity"


def main():
    df = pd.read_csv(DATA_PATH)

    # Encode categoricals
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df_enc[TARGET])

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df_enc[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("=== Severity Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))

    # Save artifacts
    joblib.dump(model, f"{MODEL_DIR}/severity_model.pkl", compress=3)
    joblib.dump(scaler, f"{MODEL_DIR}/severity_scaler.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/severity_encoders.pkl")
    joblib.dump(target_encoder, f"{MODEL_DIR}/severity_target_encoder.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/severity_feature_cols.pkl")
    print("\nSaved model + preprocessing artifacts to", MODEL_DIR)


if __name__ == "__main__":
    main()
