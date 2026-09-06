import pandas as pd
df = pd.read_csv("/home/claude/accident_project/data/accidents_features.csv")

# Cross-tab severity against a few features that "should" matter
for col in ["Driver Alcohol Level", "Speed Limit", "Weather Conditions", "Number of Fatalities"]:
    print(f"--- {col} by Accident Severity ---")
    print(df.groupby("Accident Severity")[col].mean() if df[col].dtype != object else df.groupby("Accident Severity")[col].value_counts(normalize=True))
    print()
