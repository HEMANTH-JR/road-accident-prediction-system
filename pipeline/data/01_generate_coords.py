"""
Step 1: Add synthetic latitude/longitude for each accident record.
Since the raw dataset only has Country-level location, we generate
realistic-looking coordinates within each country's approximate
bounding box. This is CLEARLY a synthetic/demo substitute for real
GPS accident coordinates and should be swapped out with real data
if this project moves past prototype stage.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# Approximate lat/long bounding boxes (min_lat, max_lat, min_lon, max_lon)
# Chosen to stay within each country's mainland extent (roughly)
COUNTRY_BBOX = {
    "USA":       (25.0, 49.0, -124.0, -67.0),
    "UK":        (50.0, 58.5, -7.5, 1.7),
    "Canada":    (43.0, 60.0, -123.0, -60.0),
    "India":     (8.0, 32.0, 70.0, 88.0),
    "China":     (21.0, 45.0, 100.0, 122.0),
    "Japan":     (31.0, 43.0, 130.0, 142.0),
    "Russia":    (45.0, 60.0, 30.0, 100.0),   # trimmed to western/populated part
    "Brazil":    (-30.0, 2.0, -60.0, -35.0),
    "Germany":   (47.5, 55.0, 6.0, 15.0),
    "Australia": (-38.0, -12.0, 115.0, 153.0),
}

def generate_coords(df: pd.DataFrame) -> pd.DataFrame:
    lats = np.empty(len(df))
    lons = np.empty(len(df))
    for country, (min_lat, max_lat, min_lon, max_lon) in COUNTRY_BBOX.items():
        mask = (df["Country"] == country).values
        n = mask.sum()
        if n == 0:
            continue
        # Cluster points around a handful of "hotspot" centers per country
        # so DBSCAN has meaningful density structure to find (rather than
        # pure uniform noise, which produces no real clusters).
        n_hotspots = np.random.randint(6, 12)
        centers_lat = np.random.uniform(min_lat, max_lat, n_hotspots)
        centers_lon = np.random.uniform(min_lon, max_lon, n_hotspots)

        assigned = np.random.randint(0, n_hotspots, n)
        jitter_lat = np.random.normal(0, (max_lat - min_lat) * 0.02, n)
        jitter_lon = np.random.normal(0, (max_lon - min_lon) * 0.02, n)

        lats[mask] = centers_lat[assigned] + jitter_lat
        lons[mask] = centers_lon[assigned] + jitter_lon

    df = df.copy()
    df["Latitude"] = lats
    df["Longitude"] = lons
    return df


if __name__ == "__main__":
    df = pd.read_csv("/mnt/user-data/uploads/road_accident_dataset.csv")
    df = generate_coords(df)
    out_path = "/home/claude/accident_project/data/accidents_with_coords.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows with coordinates -> {out_path}")
    print(df[["Country", "Latitude", "Longitude"]].groupby("Country").agg(["mean", "std"]))
