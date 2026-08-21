<div align="center">

# 🌍 Urban Heat Mitigation AI/ML System

**Optimizing urban cooling strategies via Physics-Informed AI, Geospatial Analysis, and Real-World Satellite Data.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/AI-XGBoost-orange.svg)](#)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](#)
[![Data](https://img.shields.io/badge/Data-Landsat_9_%7C_Copernicus-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

</div>

<br/>

## 📖 Overview

This project is a comprehensive geospatial AI/ML framework designed to identify urban heat stress hotspots, quantify the key drivers of urban heating, and generate optimized, scenario-based cooling interventions. It combines **Physics-Informed Machine Learning (PIML)** with advanced spatial statistics and live satellite data ingestion to provide actionable, budget-constrained cooling strategies for city planners.

## ✨ Key Features

- 🛰️ **Live Satellite Integration:** Automatically connects to Google Earth Engine and the Copernicus CDS API to fetch real Land Surface Temperature (LST), NDVI, and meteorological data.
- 🌡️ **Heat Stress Mapping:** Calculates UTCI (Universal Thermal Climate Index) to classify physiological heat stress across city grids.
- 📍 **Spatial Hotspot Detection:** Uses **Getis-Ord Gi\*** spatial statistics and Kernel Density Estimation (KDE) to accurately pinpoint heat islands.
- 🧠 **Physics-Informed ML (XGBoost):** Custom XGBoost architecture that learns the complex non-linear relationships between urban morphology and surface temperature.
- 📊 **Driver Analysis (SHAP):** Quantifies exactly *why* an area is hot by isolating the impact of LULC, Albedo, NDVI, and building morphology using game-theoretic SHAP values.
- 💡 **Cooling Optimization Engine:** A budget-constrained greedy allocator that tests interventions (Urban Greening, Cool Roofs) to maximize temperature reduction per dollar spent.
- 🖥️ **Interactive Web Dashboard:** A live **Streamlit** app for real-time scenario simulation, heat mapping, and strategy adjustments.

---

## 🚀 Quick Start

### 1. Install Dependencies

Make sure you have Python 3.9+ installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Set Up API Credentials

To download real-world data for your city, you need to authenticate with the geospatial APIs:

- **Google Earth Engine:** Run `earthengine authenticate` in your terminal to set up your local Google Cloud credentials.
- **Copernicus CDS:** Ensure you have a `.cdsapirc` file in your home directory with your Copernicus UID and API Key.

### 3. Fetch Real-World Data

Download the latest Landsat 9 and ERA5-Land data for your target city:

```bash
python src/data_pipeline/real_data_fetcher.py
```

### 4. Process the Spatial Grid

Crop the satellite imagery into a contiguous city block and extract the physical features for machine learning:

```bash
python src/data_pipeline/process_real_data.py
```

### 5. Launch the Dashboard

Open the interactive UI in your web browser to view the maps and run the AI simulations:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
urban-heat-mitigation/
├── app.py                      # Interactive Streamlit Web Dashboard
├── config.yaml                 # Master configuration (City, ML, Budget)
├── requirements.txt            # Python dependencies
├── data/
│   ├── raw/                    # Raw .tif and .nc files from APIs
│   └── processed/              # Cleaned ML-ready CSVs
├── src/                        # Core Source Code
│   ├── data_pipeline/
│   │   ├── real_data_fetcher.py   # GEE & Copernicus API integration
│   │   └── process_real_data.py   # Geospatial cropping and feature engineering
│   ├── heat_analysis/          # UTCI, Hotspot detection (Gi*), SHAP drivers
│   ├── ml_models/               # XGBoost PIML model training
│   ├── cooling_scenarios/      # Scenario simulation & Optimization Engine
│   └── visualization/          # Map rendering and charts
└── outputs/                    # Generated Assets
    └── reports/                # Final JSON/CSV scenario reports
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Machine Learning | `xgboost`, `scikit-learn` |
| Explainable AI | `shap` |
| Geospatial & Spatial Stats | `rasterio`, `xarray`, `numpy`, `pandas` |
| APIs | `earthengine-api`, `cdsapi` |
| Visualization | `matplotlib`, `streamlit` |

---

## 📄 License

This project is licensed under the [MIT License](#).

<div align="center">

*Built to mitigate urban heat islands and build sustainable cities.* 🌱

</div>
