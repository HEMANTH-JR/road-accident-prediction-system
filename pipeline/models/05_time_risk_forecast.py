"""
Step 5: Time-based Risk Forecasting.
Aggregates historical accident data into risk scores by
Hour-of-day-bucket / Day-of-week / Season, so the dashboard can show
"when" risk is highest, independent of "where" (handled by DBSCAN).
"""
import pandas as pd
import numpy as np
import joblib

DATA_PATH = "/home/claude/accident_project/data/accidents_features.csv"
MODEL_DIR = "/home/claude/accident_project/models"

SEVERITY_WEIGHT = {"Minor": 1, "Moderate": 2, "Severe": 4}


def build_time_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Severity_Weight"] = df["Accident Severity"].map(SEVERITY_WEIGHT)

    grouped = df.groupby(["Time of Day", "Day of Week", "Season"]).agg(
        Accident_Count=("Accident Severity", "count"),
        Avg_Severity_Weight=("Severity_Weight", "mean"),
        Total_Fatalities=("Number of Fatalities", "sum"),
        Total_Injuries=("Number of Injuries", "sum"),
    ).reset_index()

    # Normalize count + severity + fatalities into a single 0-100 risk score
    for col in ["Accident_Count", "Avg_Severity_Weight", "Total_Fatalities"]:
        rng = grouped[col].max() - grouped[col].min()
        grouped[f"{col}_norm"] = (grouped[col] - grouped[col].min()) / rng if rng > 0 else 0

    grouped["Risk_Score"] = (
        grouped["Accident_Count_norm"] * 40
        + grouped["Avg_Severity_Weight_norm"] * 35
        + grouped["Total_Fatalities_norm"] * 25
    ).round(1)

    grouped = grouped.sort_values("Risk_Score", ascending=False).reset_index(drop=True)
    return grouped[["Time of Day", "Day of Week", "Season", "Accident_Count",
                     "Avg_Severity_Weight", "Total_Fatalities", "Total_Injuries", "Risk_Score"]]


def build_hourly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified view: risk by Time-of-Day bucket only, for the dashboard's main chart."""
    df = df.copy()
    df["Severity_Weight"] = df["Accident Severity"].map(SEVERITY_WEIGHT)
    hourly = df.groupby("Time of Day").agg(
        Accident_Count=("Accident Severity", "count"),
        Avg_Severity_Weight=("Severity_Weight", "mean"),
        Total_Fatalities=("Number of Fatalities", "sum"),
    ).reset_index()
    hourly["Risk_Score"] = (
        (hourly["Accident_Count"] / hourly["Accident_Count"].max()) * 50
        + (hourly["Avg_Severity_Weight"] / hourly["Avg_Severity_Weight"].max()) * 50
    ).round(1)
    order = ["Morning", "Afternoon", "Evening", "Night"]
    hourly["Time of Day"] = pd.Categorical(hourly["Time of Day"], categories=order, ordered=True)
    return hourly.sort_values("Time of Day").reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    time_risk = build_time_risk_table(df)
    print("Top 10 highest-risk time/day/season combinations:")
    print(time_risk.head(10).to_string(index=False))

    hourly = build_hourly_summary(df)
    print("\nRisk by Time of Day:")
    print(hourly.to_string(index=False))

    time_risk.to_csv("/home/claude/accident_project/data/time_risk_table.csv", index=False)
    hourly.to_csv("/home/claude/accident_project/data/hourly_risk_summary.csv", index=False)
    print("\nSaved time_risk_table.csv and hourly_risk_summary.csv")
