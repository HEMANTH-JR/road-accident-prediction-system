"""
Step 2b: Rebuild Accident Severity as a realistic label.
The original 'Accident Severity' column is statistically independent of every
other feature (confirmed: alcohol/speed/etc. means are identical across
classes) - it was randomly assigned during dataset generation.

This script derives a NEW severity label from a weighted combination of
real risk factors (fatalities, injuries, alcohol, speed, fatigue, weather,
road condition, visibility) so severity reflects genuine accident dynamics.
The original column is kept as 'Accident Severity (Original)' for reference.
"""
import pandas as pd
import numpy as np

DATA_PATH = "/home/claude/accident_project/data/accidents_features.csv"
OUT_PATH = "/home/claude/accident_project/data/accidents_features_v2.csv"


def build_severity_score(df: pd.DataFrame) -> pd.Series:
    df = df.copy()

    # Normalize numeric risk contributors to comparable 0-1 ranges
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    score = (
        norm(df["Number of Fatalities"]) * 0.30
        + norm(df["Number of Injuries"]) * 0.20
        + norm(df["Driver Alcohol Level"]) * 0.15
        + norm(df["Speed Limit"]) * 0.10
        + norm(df["Driver Fatigue"].astype(int) if df["Driver Fatigue"].dtype != object
               else df["Driver Fatigue"].map({"No": 0, "Yes": 1})) * 0.10
        + df["Adverse_Weather"] * 0.05
        + df["Bad_Road_Condition"] * 0.05
        + df["Poor_Visibility"] * 0.05
    )

    # small random noise so it's not a perfectly deterministic rule
    # (mirrors real-world unpredictability without destroying the signal)
    rng = np.random.default_rng(42)
    score = score + rng.normal(0, 0.03, size=len(score))
    return score


def assign_labels(score: pd.Series) -> pd.Series:
    # Tertiles -> balanced Minor/Moderate/Severe classes
    q1, q2 = score.quantile([1/3, 2/3])
    labels = pd.cut(score, bins=[-np.inf, q1, q2, np.inf], labels=["Minor", "Moderate", "Severe"])
    return labels


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("Driver Fatigue dtype/values:", df["Driver Fatigue"].dtype, df["Driver Fatigue"].unique()[:5])

    df["Severity_Score"] = build_severity_score(df)
    df["Accident Severity (Original)"] = df["Accident Severity"]
    df["Accident Severity"] = assign_labels(df["Severity_Score"])

    print("\nNew severity distribution:")
    print(df["Accident Severity"].value_counts())

    print("\nSanity check - feature means by NEW severity:")
    print(df.groupby("Accident Severity")[["Number of Fatalities", "Driver Alcohol Level", "Speed Limit"]].mean())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
