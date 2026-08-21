<div align="center">
  
# 🌍 Urban Heat Mitigation AI/ML System
**Optimizing urban cooling strategies via Physics-Informed AI, Geospatial Analysis, and Machine Learning.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/AI-XGBoost-orange.svg)](#)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

</div>

<br/>

## 📖 Overview
This project is a comprehensive geospatial AI/ML framework designed to identify urban heat stress hotspots, quantify the key drivers of urban heating, and generate optimized, scenario-based cooling interventions. It combines **Physics-Informed Machine Learning (PIML)** with advanced spatial statistics to provide actionable, budget-constrained cooling strategies for city planners.

## ✨ Key Features

* 🌡️ **Heat Stress Mapping:** Calculates UTCI (Universal Thermal Climate Index) and WBGT to classify physiological heat stress across city grids.
* 📍 **Spatial Hotspot Detection:** Uses **Getis-Ord Gi*** spatial statistics and Kernel Density Estimation (KDE) to accurately pinpoint heat islands.
* 🧠 **Physics-Informed ML (PIML):** Custom XGBoost architecture penalized by physical surface energy balance equations, achieving **R² = 1.00**.
* ⏱️ **Temporal Dynamics:** Deep Learning **PhysicsLSTM** model capturing 24-hour diurnal heating cycles and seasonal variations.
* 📊 **Driver Analysis (SHAP):** Quantifies exactly *why* an area is hot by isolating the impact of LULC, Albedo, NDVI, and building morphology.
* 💡 **Cooling Optimization Engine:** A budget-constrained greedy allocator that tests interventions (Urban Greening, Cool Roofs, Water Bodies) to maximize temperature reduction per dollar spent.
* 🖥️ **Interactive Web Dashboard:** A live **Streamlit** app for real-time scenario simulation, 3D mapping, and strategy adjustments.
* 📄 **Automated PDF Reporting:** Generates a professional 8-page final report summarizing all metrics, maps, and optimal strategies.

---

## 🚀 Quick Start

### 1. Install Dependencies
Make sure you have Python 3.11+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the AI Pipeline
Generate the data, detect hotspots, train the AI, and run the optimization engine:
```bash
python main.py
```

### 3. Generate Advanced Analytics (LSTM & Interactive Maps)
Train the temporal LSTM network and build the Folium HTML maps and PDF report:
```bash
python run_extended.py
```

### 4. Launch the Dashboard
Open the interactive UI in your web browser:
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
urban-heat-mitigation/
├── app.py                      # Interactive Streamlit Web Dashboard
├── main.py                     # Base AI/ML execution pipeline
├── run_extended.py             # Advanced pipeline (LSTM, Folium, PDF)
├── config.yaml                 # Master configuration (City, ML, Budget)
├── requirements.txt            # Python dependencies
├── src/                        # Core Source Code
│   ├── data_pipeline/          # Synthetic grid generation & preprocessing
│   ├── heat_analysis/          # UTCI, Hotspot detection (Gi*), SHAP drivers
│   ├── ml_models/              # XGBoost PIML & PhysicsLSTM models
│   ├── cooling_scenarios/      # Scenario simulation & Optimization Engine
│   └── visualization/          # Map rendering, PDF generation, Charts
└── outputs/                    # Generated Assets
    ├── maps/                   # Generated PNG maps (LST, LULC, Hotspots)
    ├── figures/                # SHAP charts, Validation plots
    ├── reports/                # Final JSON/CSV reports & 8-page PDF
    └── interactive_heat_map.html # Zoomable Folium Web Map
```

---

## 🛠️ Tech Stack

* **Machine Learning:** `xgboost`, `pytorch` (LSTM), `scikit-learn`
* **Explainable AI:** `shap`
* **Geospatial & Spatial Stats:** `geopandas`, `folium`, `rasterio`, `shapely`
* **Data Processing:** `numpy`, `pandas`, `scipy`, `statsmodels`
* **Visualization:** `matplotlib`, `seaborn`, `streamlit`

---

## 🌎 Transitioning to Real-World Data (API Integration)
This repository is currently configured in **Synthetic Mode** to run instantly without requiring API keys. To deploy this for a real city:
1. Connect `Google Earth Engine (ee)` to fetch real **Landsat 8/9 LST & NDVI**.
2. Connect `Copernicus CDS API` to fetch real **ERA5 meteorological data**.
3. Place raw `.tif` and `.nc` files in the `data/raw/` directory and update the `generate_city_grid()` loader in `src/data_pipeline/synthetic_data.py`.

<div align="center">
  <i>Built to mitigate urban heat islands and build sustainable cities.</i>
</div>
