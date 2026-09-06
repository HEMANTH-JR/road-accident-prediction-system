"""
Step 4: DBSCAN Black-Spot Detection.
Clusters accident coordinates to find dense "black spot" zones.
Run per-country since coordinate scales/hotspot structure differ by country.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN

DATA_PATH = "/home/claude/accident_project/data/accidents_features.csv"
MODEL_DIR = "/home/claude/accident_project/models"

# DBSCAN params tuned for degrees lat/long (~ roughly 5-10km radius depending on latitude)
EPS_DEGREES = 0.15
MIN_SAMPLES = 25


def run_dbscan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Cluster"] = -2  # sentinel, overwritten per country

    for country, group in df.groupby("Country"):
        coords = group[["Latitude", "Longitude"]].values
        db = DBSCAN(eps=EPS_DEGREES, min_samples=MIN_SAMPLES).fit(coords)
        df.loc[group.index, "Cluster"] = db.labels_

    # Global cluster id so cluster numbers don't collide across countries
    df["Black_Spot_ID"] = df["Country"].astype(str) + "_" + df["Cluster"].astype(str)
    df.loc[df["Cluster"] == -1, "Black_Spot_ID"] = "NOISE"

    return df


def summarize_blackspots(df: pd.DataFrame) -> pd.DataFrame:
    clustered = df[df["Cluster"] != -1]
    summary = clustered.groupby("Black_Spot_ID").agg(
        Country=("Country", "first"),
        Num_Accidents=("Cluster", "count"),
        Center_Lat=("Latitude", "mean"),
        Center_Lon=("Longitude", "mean"),
        Severe_Count=("Accident Severity", lambda s: (s == "Severe").sum()),
        Total_Fatalities=("Number of Fatalities", "sum"),
        Total_Injuries=("Number of Injuries", "sum"),
    ).reset_index()
    summary["Risk_Score"] = (
        summary["Num_Accidents"] * 0.4
        + summary["Total_Fatalities"] * 3
        + summary["Total_Injuries"] * 1
        + summary["Severe_Count"] * 2
    )
    summary = summary.sort_values("Risk_Score", ascending=False).reset_index(drop=True)
    return summary


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df = run_dbscan(df)

    n_clusters = df.loc[df["Cluster"] != -1, "Black_Spot_ID"].nunique()
    n_noise = (df["Cluster"] == -1).sum()
    print(f"Found {n_clusters} black-spot clusters across all countries")
    print(f"Noise points (not in any cluster): {n_noise} ({n_noise/len(df)*100:.1f}%)")

    summary = summarize_blackspots(df)
    print("\nTop 10 highest-risk black spots:")
    print(summary.head(10).to_string(index=False))

    out_path = "/home/claude/accident_project/data/accidents_clustered.csv"
    df.to_csv(out_path, index=False)
    summary_path = "/home/claude/accident_project/data/blackspot_summary.csv"
    summary.to_csv(summary_path, index=False)
    joblib.dump({"eps": EPS_DEGREES, "min_samples": MIN_SAMPLES}, f"{MODEL_DIR}/dbscan_params.pkl")
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {summary_path}")
