// ==========================================================================
// Road Risk Command — dashboard logic
// ==========================================================================

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`Request failed: ${url}`);
  return res.json();
}

function animateCount(el, target, suffix = "", duration = 900) {
  const start = 0;
  const startTime = performance.now();
  function tick(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = Math.round(start + (target - start) * eased);
    el.textContent = value.toLocaleString() + suffix;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ---- KPI strip ----
async function loadKPIs() {
  const kpis = await fetchJSON("/api/kpis");
  const strip = document.getElementById("kpi-strip");

  const items = [
    { label: "Accidents Analyzed", value: kpis.total_accidents, cls: "" },
    { label: "Total Fatalities", value: kpis.total_fatalities, cls: "red" },
    { label: "Total Injuries", value: kpis.total_injuries, cls: "amber" },
    { label: "Black Spots Found", value: kpis.black_spots_found, cls: "amber" },
    { label: "Countries Covered", value: kpis.countries_covered, cls: "" },
    { label: "Highest Risk Window", value: kpis.highest_risk_time, cls: "red", text: true },
  ];

  strip.innerHTML = items.map((item, i) => `
    <div class="kpi">
      <div class="kpi-label">${item.label}</div>
      <div class="kpi-value ${item.cls}" id="kpi-${i}">${item.text ? item.value : "0"}</div>
    </div>
  `).join("");

  items.forEach((item, i) => {
    if (!item.text) animateCount(document.getElementById(`kpi-${i}`), item.value);
  });
}

// ---- Map + heatmap + blackspot markers ----
async function loadMap() {
  const map = L.map("risk-map", { worldCopyJump: true }).setView([20, 10], 2);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap &copy; CARTO",
    maxZoom: 18,
  }).addTo(map);

  const points = await fetchJSON("/api/heatmap");
  const heatPoints = points.map(p => [p[0], p[1], p[2]]);
  L.heatLayer(heatPoints, {
    radius: 18,
    blur: 22,
    maxZoom: 6,
    gradient: { 0.2: "#2e9e6d", 0.5: "#f5a623", 0.85: "#e0393e" },
  }).addTo(map);

  const blackspots = await fetchJSON("/api/blackspots");
  blackspots.forEach(bs => {
    const marker = L.circleMarker([bs.Center_Lat, bs.Center_Lon], {
      radius: 6,
      color: "#f5c542",
      weight: 1.5,
      fillColor: "#e0393e",
      fillOpacity: 0.85,
    }).addTo(map);

    marker.bindPopup(`
      <strong>${bs.Black_Spot_ID}</strong><br>
      ${bs.Country}<br>
      Accidents: ${bs.Num_Accidents}<br>
      Fatalities: ${bs.Total_Fatalities}<br>
      Risk score: ${Math.round(bs.Risk_Score).toLocaleString()}
    `);
  });

  return blackspots;
}

// ---- Black spot ranked list ----
function renderBlackspotList(blackspots) {
  const container = document.getElementById("blackspot-list");
  const maxRisk = Math.max(...blackspots.map(b => b.Risk_Score));

  container.innerHTML = blackspots.slice(0, 10).map((bs, i) => `
    <div class="blackspot-row">
      <span class="rank">${String(i + 1).padStart(2, "0")}</span>
      <div class="blackspot-info">
        <span class="blackspot-name">${bs.Black_Spot_ID}</span>
        <span class="blackspot-meta">${bs.Country} · ${bs.Num_Accidents} accidents · ${bs.Total_Fatalities} fatalities</span>
      </div>
      <div class="risk-bar-track">
        <div class="risk-bar-fill" style="width:${(bs.Risk_Score / maxRisk * 100).toFixed(0)}%"></div>
      </div>
    </div>
  `).join("");
}

// ---- Time-of-day risk chart ----
async function loadTimeRisk() {
  const data = await fetchJSON("/api/time-risk");
  const hourly = data.hourly;

  const ctx = document.getElementById("time-chart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: hourly.map(h => h["Time of Day"]),
      datasets: [{
        data: hourly.map(h => h.Risk_Score),
        backgroundColor: hourly.map(h =>
          h["Time of Day"] === "Night" ? "#e0393e" : "#f5a623"
        ),
        borderRadius: 4,
        maxBarThickness: 48,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8a93a6", font: { family: "IBM Plex Mono", size: 11 } }, grid: { display: false } },
        y: { ticks: { color: "#8a93a6", font: { family: "IBM Plex Mono", size: 10 } }, grid: { color: "#2e3440" }, beginAtZero: true },
      },
    },
  });

  const top = data.top_combos[0];
  const note = document.getElementById("time-note");
  note.innerHTML = `
    Highest-risk window overall: <strong>${top["Time of Day"]}, ${top["Day of Week"]}, ${top["Season"]}</strong>
    — risk score ${top.Risk_Score}/100 based on accident volume, severity, and fatalities.<br><br>
    Night consistently scores highest across all regions, suggesting patrol and lighting
    resources are best weighted toward late-night hours.
  `;
}

// ---- Prediction tool ----
function setupPredictForm() {
  const btn = document.getElementById("predict-btn");
  btn.addEventListener("click", async () => {
    btn.textContent = "Predicting…";
    btn.disabled = true;

    const payload = {
      road_type: document.getElementById("road_type").value,
      weather: document.getElementById("weather").value,
      road_condition: document.getElementById("road_condition").value,
      time_of_day: document.getElementById("time_of_day").value,
      speed_limit: document.getElementById("speed_limit").value,
      alcohol_level: document.getElementById("alcohol_level").value,
      fatigue: document.getElementById("fatigue").value,
      visibility: document.getElementById("visibility").value,
    };

    try {
      const result = await fetchJSON("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const badge = document.getElementById("severity-badge");
      badge.textContent = result.severity;
      badge.className = `severity-badge ${result.severity}`;

      document.getElementById("confidence-text").textContent =
        `Confidence: ${result.confidence}% · Minor ${result.probabilities.Minor}% · Moderate ${result.probabilities.Moderate}% · Severe ${result.probabilities.Severe}%`;

      document.getElementById("predict-result").classList.add("show");
    } catch (e) {
      alert("Prediction failed — check the server console.");
    } finally {
      btn.textContent = "Predict Severity";
      btn.disabled = false;
    }
  });
}

// ---- Init ----
(async function init() {
  loadKPIs();
  setupPredictForm();
  const blackspots = await loadMap();
  renderBlackspotList(blackspots);
  loadTimeRisk();
})();
