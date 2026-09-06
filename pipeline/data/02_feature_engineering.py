"""
Step 2: Feature engineering.
Builds spatial + temporal features used by all downstream models.
"""
import pandas as pd
import numpy as np

MONTH_ORDER = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

MONTH_TO_SEASON = {
    "December": "Winter", "January": "Winter", "February": "Winter",
    "March": "Spring", "April": "Spring", "May": "Spring",
    "June": "Summer", "July": "Summer", "August": "Summer",
    "September": "Autumn", "October": "Autumn", "November": "Autumn",
}

TIME_TO_HOUR = {  # rough midpoint hour for each bucket, used for cyclic encoding
    "Morning": 8, "Afternoon": 14, "Evening": 19, "Night": 1,
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Temporal features ---
    df["Season"] = df["Month"].map(MONTH_TO_SEASON)
    df["Month_Num"] = df["Month"].map({m: i+1 for i, m in enumerate(MONTH_ORDER)})
    df["Approx_Hour"] = df["Time of Day"].map(TIME_TO_HOUR)

    # cyclic encodings so ML models see time as circular, not linear
    df["Month_sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Approx_Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Approx_Hour"] / 24)

    df["Is_Weekend"] = df["Day of Week"].isin(["Saturday", "Sunday"]).astype(int)
    df["Is_Night"] = (df["Time of Day"] == "Night").astype(int)

    # --- Risk-related engineered features ---
    df["People_Involved"] = (
        df["Number of Vehicles Involved"] + df["Pedestrians Involved"] + df["Cyclists Involved"]
    )
    df["Casualty_Rate"] = (df["Number of Injuries"] + df["Number of Fatalities"]) / df["People_Involved"].replace(0, 1)
    df["High_Alcohol"] = (df["Driver Alcohol Level"] > 0.08).astype(int)  # common legal threshold
    df["Poor_Visibility"] = (df["Visibility Level"] < df["Visibility Level"].quantile(0.25)).astype(int)
    df["Adverse_Weather"] = df["Weather Conditions"].isin(["Rainy", "Snowy", "Foggy", "Windy"]).astype(int)
    df["Bad_Road_Condition"] = df["Road Condition"].isin(["Wet", "Icy", "Snow-covered"]).astype(int)

    return df


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/accident_project/data/accidents_with_coords.csv")
    df = engineer_features(df)
    out_path = "/home/claude/accident_project/data/accidents_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved engineered dataset -> {out_path}")
    print(df.shape)
    print(df[["Season","Is_Weekend","Is_Night","Casualty_Rate","High_Alcohol",
              "Poor_Visibility","Adverse_Weather","Bad_Road_Condition"]].describe(include="all"))
