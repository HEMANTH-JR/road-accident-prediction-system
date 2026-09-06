# Road Accident Prediction System — Dashboard

ML + GIS prototype for proactive traffic safety management. Combines a
RandomForest severity classifier, DBSCAN black-spot clustering, and
time-based risk aggregation, all served through a Flask dashboard with
an interactive GIS heatmap.

## What's inside

```
webapp/
├── app.py                  # Flask backend + live prediction endpoint
├── requirements.txt
├── models/                 # Trained model + preprocessing artifacts
├── static/
│   ├── css/style.css
│   ├── js/dashboard.js
│   └── data/               # Precomputed JSON (heatmap, blackspots, KPIs, time-risk)
└── templates/
    └── index.html
```

Pipeline scripts that produced the data/models (kept for reference /
re-running if you get real data):

```
data/
├── 01_generate_coords.py       # synthetic lat/long per country
├── 02_feature_engineering.py   # spatial + temporal features
├── 02b_rebuild_severity.py     # re-derives severity as a learnable label
└── 06_export_dashboard_data.py # exports JSON for the dashboard

models/
├── 03_train_severity_model.py  # RandomForest severity classifier
├── 04_dbscan_blackspots.py     # DBSCAN black-spot detection
└── 05_time_risk_forecast.py    # hour/day/season risk aggregation
```

## Running it

```bash
cd webapp
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

## What you'll see

- **KPI strip** — total accidents, fatalities, injuries, black spots found, countries covered
- **GIS heatmap** — accident density/severity heatmap with markers on the top 30 black spots (click for details)
- **Top black spots** — ranked list by risk score (accident count + fatalities + severity weighted)
- **Risk by time of day** — bar chart showing Night as consistently highest-risk
- **Severity prediction tool** — enter road/weather/driver conditions and get a live Minor/Moderate/Severe prediction with confidence scores, straight from the trained model

## Important notes for your report/demo

1. **Coordinates are synthetic.** The source dataset only had Country-level
   location, not GPS coordinates. `01_generate_coords.py` generates
   realistic per-country hotspot coordinates so DBSCAN has real spatial
   structure to cluster. Swap in real accident GPS data if you get access
   to it — the rest of the pipeline doesn't need to change.

2. **Severity labels were re-derived.** The original "Accident Severity"
   column in the raw dataset was statistically independent of every other
   field (verified: alcohol level, speed, etc. had identical means across
   Minor/Moderate/Severe) — it was randomly assigned. `02b_rebuild_severity.py`
   builds a new severity label from a weighted combination of real risk
   factors (fatalities, injuries, alcohol, speed, fatigue, weather, road
   condition, visibility) so the classifier has genuine signal to learn.
   This is disclosed in the code comments — be upfront about it in your
   submission too.

3. **Model performance:** ~49% accuracy / 0.47 macro F1 on the 3-class
   severity task (vs. 33% random baseline). Reasonable for a synthetic-label
   demo; Minor and Severe are predicted well, Moderate is the harder middle
   class (expected in ordinal problems like this).

4. **To re-run the full pipeline** with a new dataset: replace the CSV in
   `data/`, run scripts 01 → 02 → 02b → 03 → 04 → 05 → 06 in order, then
   copy the new model artifacts into `webapp/models/` and JSON into
   `webapp/static/data/`.
