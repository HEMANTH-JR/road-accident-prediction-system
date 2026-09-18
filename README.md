[README (1).md](https://github.com/user-attachments/files/32367205/README.1.md)
# 🚦 Road Risk Command

An interactive dashboard for analyzing road accident data — surfacing high-risk locations, dangerous time windows, and predicting accident severity from live inputs.

![status](https://img.shields.io/badge/status-active-brightgreen) ![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

Road Risk Command ingests historical road accident data and turns it into an actionable, real-time command center. It combines geospatial risk mapping, black-spot ranking, temporal risk analysis, and a machine-learning-backed severity predictor in a single dark-themed dashboard.

## Features

- **KPI Strip** — animated at-a-glance counters for accidents analyzed, fatalities, injuries, black spots found, countries covered, and the highest-risk time window.
- **Risk Map** — a dark-tiled Leaflet map with:
  - A heatmap layer showing accident density and severity.
  - Circle markers for each identified black spot, with popups showing accident count, fatalities, and computed risk score.
- **Black Spot Rankings** — top 10 highest-risk locations ranked by risk score, with proportional risk bars.
- **Time-of-Day Risk Chart** — a bar chart (Chart.js) breaking down risk score by time of day, highlighting night-time risk, plus a call-out for the single highest-risk combination of time, day, and season.
- **Severity Predictor** — a form-driven tool that sends road/weather/driver conditions to a prediction endpoint and returns a severity classification (Minor / Moderate / Severe) with confidence and per-class probabilities.

## Tech Stack

- **Frontend:** Vanilla JavaScript, [Leaflet.js](https://leafletjs.com/) + Leaflet.heat for mapping, [Chart.js](https://www.chartjs.org/) for charts
- **Backend:** REST API (see [API Reference](#api-reference)) — swap in your actual stack here (e.g. Flask / FastAPI / Node)
- **ML Model:** Severity classification model served via `/api/predict` — note your model type/framework here

## Project Structure

```
.
├── dashboard.js        # Frontend dashboard logic (KPIs, map, charts, predictor)
├── index.html           # Dashboard markup (add if present)
├── styles.css            # Dashboard styling (add if present)
├── server/                # Backend API + model serving (add your details)
└── data/                    # Source accident dataset(s) (add your details)
```

> Adjust this tree to match your actual repo layout.

## API Reference

The dashboard expects the following endpoints from the backend:

| Endpoint | Method | Description |
|---|---|---|
| `/api/kpis` | GET | Returns summary KPIs: `total_accidents`, `total_fatalities`, `total_injuries`, `black_spots_found`, `countries_covered`, `highest_risk_time` |
| `/api/heatmap` | GET | Returns an array of `[lat, lon, intensity]` points for the heatmap layer |
| `/api/blackspots` | GET | Returns black spot records with `Black_Spot_ID`, `Country`, `Center_Lat`, `Center_Lon`, `Num_Accidents`, `Total_Fatalities`, `Risk_Score` |
| `/api/time-risk` | GET | Returns `hourly` (risk score per time-of-day bucket) and `top_combos` (highest-risk time/day/season combinations) |
| `/api/predict` | POST | Accepts `road_type`, `weather`, `road_condition`, `time_of_day`, `speed_limit`, `alcohol_level`, `fatigue`, `visibility`; returns `severity`, `confidence`, and `probabilities` (Minor/Moderate/Severe) |

## Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/road-risk-command.git
cd road-risk-command

# Install backend dependencies
# (fill in based on your stack, e.g.)
pip install -r requirements.txt

# Start the backend server
# python app.py

# Open the dashboard
# Serve index.html + dashboard.js via your backend or a static file server
```

> Replace the install/run commands above with the ones that match your actual backend.

## Dataset

Describe the accident dataset you used here — source, time range, geographic coverage, and any preprocessing steps applied to compute black spots and risk scores.

## Roadmap

- [ ] Add authentication for the prediction API
- [ ] Support filtering the map/charts by date range or country
- [ ] Export black spot reports as PDF/CSV
- [ ] Model retraining pipeline

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
