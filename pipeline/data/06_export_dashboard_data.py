"""
Step 6: Export lightweight aggregated JSON files for the web dashboard.
Raw 132k-row CSV is too heavy to ship to the browser directly, so we
aggregate accident points into a coordinate grid (for the heatmap layer)
and precompute summary tables the dashboard reads on load.
"""
import pandas as pd
import numpy as np
import json

DATA_DIR = "/home/claude/accident_project/data"
OUT_DIR = "/home/claude/accident_project/webapp/static/data"

import os
os.makedirs(OUT_DIR, exist_ok=True)


def export_heatmap_points(df: pd.DataFrame):
    # Round coordinates to ~0.1 degree grid (~11km) and count accidents per cell
    # This keeps the heatmap payload small while preserving hotspot shape.
    grid = df.copy()
    grid["lat_r"] = grid["Latitude"].round(1)
    grid["lon_r"] = grid["Longitude"].round(1)
    agg = grid.groupby(["lat_r", "lon_r"]).agg(
        count=("Latitude", "count"),
        fatalities=("Number of Fatalities", "sum"),
    ).reset_index()
    agg["weight"] = agg["count"] + agg["fatalities"] * 2
    max_w = agg["weight"].max()
    agg["weight_norm"] = (agg["weight"] / max_w).round(3)

    points = agg[["lat_r", "lon_r", "weight_norm"]].values.tolist()
    with open(f"{OUT_DIR}/heatmap_points.json", "w") as f:
        json.dump(points, f)
    print(f"heatmap_points.json: {len(points)} grid cells")


def export_blackspots():
    summary = pd.read_csv(f"{DATA_DIR}/blackspot_summary.csv")
    top = summary.head(30)
    records = top.to_dict(orient="records")
    with open(f"{OUT_DIR}/blackspots.json", "w") as f:
        json.dump(records, f)
    print(f"blackspots.json: {len(records)} top black spots")


def export_time_risk():
    hourly = pd.read_csv(f"{DATA_DIR}/hourly_risk_summary.csv")
    combo = pd.read_csv(f"{DATA_DIR}/time_risk_table.csv")
    payload = {
        "hourly": hourly.to_dict(orient="records"),
        "top_combos": combo.head(10).to_dict(orient="records"),
    }
    with open(f"{OUT_DIR}/time_risk.json", "w") as f:
        json.dump(payload, f)
    print("time_risk.json exported")


def export_kpis(df: pd.DataFrame):
    blackspots = pd.read_csv(f"{DATA_DIR}/blackspot_summary.csv")
    kpis = {
        "total_accidents": int(len(df)),
        "total_fatalities": int(df["Number of Fatalities"].sum()),
        "total_injuries": int(df["Number of Injuries"].sum()),
        "countries_covered": int(df["Country"].nunique()),
        "black_spots_found": int(len(blackspots)),
        "severe_pct": round((df["Accident Severity"] == "Severe").mean() * 100, 1),
        "highest_risk_time": "Night",
        "highest_risk_blackspot": blackspots.sort_values("Risk_Score", ascending=False).iloc[0]["Black_Spot_ID"],
    }
    with open(f"{OUT_DIR}/kpis.json", "w") as f:
        json.dump(kpis, f)
    print("kpis.json:", kpis)


def export_country_bounds():
    """So the map can fly to a sensible default view."""
    df = pd.read_csv(f"{DATA_DIR}/accidents_features_v2.csv")
    bounds = df.groupby("Country").agg(
        lat=("Latitude", "mean"), lon=("Longitude", "mean")
    ).reset_index().to_dict(orient="records")
    with open(f"{OUT_DIR}/country_centers.json", "w") as f:
        json.dump(bounds, f)


if __name__ == "__main__":
    df = pd.read_csv(f"{DATA_DIR}/accidents_features_v2.csv")
    export_heatmap_points(df)
    export_blackspots()
    export_time_risk()
    export_kpis(df)
    export_country_bounds()
    print("\nAll dashboard data exported to", OUT_DIR)
