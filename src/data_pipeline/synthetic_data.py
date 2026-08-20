"""
synthetic_data.py
-----------------
Generates physics-grounded synthetic urban data mimicking a large Indian city
(e.g., Delhi). All outputs are spatially correlated and physically consistent.

Swap-in point: Replace `generate_city_grid()` with real GEE/ERA5 API calls
to load actual Landsat LST, NDVI, LULC, and ERA5 atmospheric data.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import yaml
import os


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _spatial_noise(rows: int, cols: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Generate spatially correlated noise using Gaussian smoothing."""
    noise = rng.standard_normal((rows, cols))
    return gaussian_filter(noise, sigma=sigma)


def _normalize(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize array to [vmin, vmax]."""
    a = arr - arr.min()
    a = a / (a.max() + 1e-10)
    return a * (vmax - vmin) + vmin


def generate_city_grid(config: dict) -> dict:
    """
    Generate a synthetic city dataset as a dictionary of 2D arrays.

    Returns
    -------
    dict with keys:
        lulc, lst, ndvi, albedo, svf (sky view factor), 
        building_height, impervious_fraction,
        air_temp, humidity, wind_speed,
        latitude_grid, longitude_grid
    """
    g = config["grid"]
    rows, cols = g["rows"], g["cols"]
    seed = g["seed"]
    rng = np.random.default_rng(seed)

    d = config["data"]
    lc = config["lulc"]
    city = config["city"]

    print(f"  Generating {rows}×{cols} synthetic grid for {city['name']}...")

    # ── Urban structure: dense core, suburban ring, peripheral areas ──────────
    y_idx, x_idx = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij")
    dist_from_center = np.sqrt(y_idx**2 + x_idx**2)  # 0=center, ~1.4=corner

    # ── Land Use / Land Cover ─────────────────────────────────────────────────
    # Urban core: high-density commercial/residential
    # Mid ring: medium-density residential
    # Outskirts: sparse vegetation, agriculture
    lulc_base = np.zeros((rows, cols), dtype=int)
    lulc_noise = _spatial_noise(rows, cols, sigma=8, rng=rng)

    # Distance-based LULC assignment with noise
    fractions = lc["fractions"]
    lulc_base[dist_from_center < 0.15] = 6   # Commercial/Industrial core
    lulc_base[(dist_from_center >= 0.15) & (dist_from_center < 0.35)] = 5  # High-density residential
    lulc_base[(dist_from_center >= 0.35) & (dist_from_center < 0.55)] = 4  # Low-density residential
    lulc_base[(dist_from_center >= 0.55) & (dist_from_center < 0.70)] = 2  # Sparse vegetation
    lulc_base[(dist_from_center >= 0.70) & (dist_from_center < 0.85)] = 3  # Agriculture
    lulc_base[dist_from_center >= 0.85] = 1  # Dense vegetation/forest

    # Add water bodies (rivers/lakes) – sinusoidal paths
    river_mask = np.abs(y_idx - 0.3 * np.sin(3 * x_idx)) < 0.04
    lake_mask = (dist_from_center > 0.4) & (dist_from_center < 0.48) & (x_idx > 0.2) & (x_idx < 0.5)
    water_mask = river_mask | lake_mask
    lulc_base[water_mask] = 0

    # Construction/Barren patches
    barren_patches = (lulc_noise > 1.5) & (dist_from_center > 0.3) & (dist_from_center < 0.7)
    lulc_base[barren_patches] = 7

    lulc = lulc_base

    # ── NDVI – inversely related to urban density ─────────────────────────────
    lulc_to_ndvi = {0: 0.0, 1: 0.75, 2: 0.55, 3: 0.45, 4: 0.25, 5: 0.10, 6: 0.05, 7: -0.02}
    ndvi_base = np.vectorize(lulc_to_ndvi.get)(lulc)
    ndvi_noise = _spatial_noise(rows, cols, sigma=5, rng=rng) * 0.08
    ndvi = np.clip(ndvi_base + ndvi_noise, d["ndvi_range"][0], d["ndvi_range"][1])

    # ── Albedo – high for roofs/pavement, low for vegetation/water ────────────
    lulc_to_albedo = {0: 0.06, 1: 0.12, 2: 0.15, 3: 0.18, 4: 0.22, 5: 0.28, 6: 0.35, 7: 0.30}
    albedo_base = np.vectorize(lulc_to_albedo.get)(lulc)
    albedo_noise = _spatial_noise(rows, cols, sigma=4, rng=rng) * 0.04
    albedo = np.clip(albedo_base + albedo_noise, d["albedo_range"][0], d["albedo_range"][1])

    # ── Building height → Sky View Factor (SVF) ───────────────────────────────
    lulc_to_bh = {0: 0, 1: 2, 2: 3, 3: 1, 4: 8, 5: 18, 6: 30, 7: 0}
    bh_base = np.vectorize(lulc_to_bh.get)(lulc).astype(float)
    bh_noise = _spatial_noise(rows, cols, sigma=6, rng=rng) * 5
    building_height = np.clip(bh_base + bh_noise, 0, d["building_height_range"][1])

    # SVF ≈ 1 for open areas, lower in dense urban canyons
    svf = np.clip(1.0 - (building_height / 80.0) * 0.6, 0.2, 1.0)

    # ── Impervious Surface Fraction ───────────────────────────────────────────
    lulc_to_imp = {0: 0.0, 1: 0.05, 2: 0.10, 3: 0.05, 4: 0.45, 5: 0.75, 6: 0.90, 7: 0.60}
    impervious_fraction = np.vectorize(lulc_to_imp.get)(lulc).astype(float)
    imp_noise = _spatial_noise(rows, cols, sigma=5, rng=rng) * 0.06
    impervious_fraction = np.clip(impervious_fraction + imp_noise, 0.0, 1.0)

    # ── ERA5 Atmospheric Variables ────────────────────────────────────────────
    # Air temperature: slightly lower at periphery (rural cooling effect)
    air_temp_base = _normalize(0.7 - dist_from_center, d["air_temp_range"][0], d["air_temp_range"][1])
    air_temp_noise = _spatial_noise(rows, cols, sigma=12, rng=rng) * 1.5
    air_temp = np.clip(air_temp_base + air_temp_noise,
                       d["air_temp_range"][0], d["air_temp_range"][1])

    # Humidity: higher near water, lower in dense urban
    humidity_base = 60.0 - 15.0 * impervious_fraction
    humidity_base[water_mask] = 85.0
    humidity_noise = _spatial_noise(rows, cols, sigma=10, rng=rng) * 8
    humidity = np.clip(humidity_base + humidity_noise,
                       d["humidity_range"][0], d["humidity_range"][1])

    # Wind speed: reduced in dense urban canyons
    wind_base = d["wind_speed_range"][1] * svf
    wind_noise = _spatial_noise(rows, cols, sigma=8, rng=rng) * 1.0
    wind_speed = np.clip(wind_base + wind_noise,
                         d["wind_speed_range"][0], d["wind_speed_range"][1])

    # ── Land Surface Temperature (physics-informed) ───────────────────────────
    # LST = f(NDVI, albedo, impervious, SVF, air_temp)
    # Physical relationships:
    #   - Higher NDVI → lower LST (evapotranspiration cooling)
    #   - Higher albedo → lower LST (less absorbed solar)
    #   - Higher impervious fraction → higher LST (heat storage)
    #   - Lower SVF → trapping of longwave radiation → higher LST
    #   - Higher air temp → higher LST
    lst_base = (
        air_temp
        + 8.0 * impervious_fraction       # Impervious heat storage
        - 12.0 * ndvi                      # Vegetation cooling
        - 6.0 * albedo                     # Albedo cooling
        + 4.0 * (1.0 - svf)               # Urban canyon trapping
        + _spatial_noise(rows, cols, sigma=7, rng=rng) * 2.0  # residual noise
    )
    lst = np.clip(lst_base, d["lst_range_celsius"][0], d["lst_range_celsius"][1])
    lst[water_mask] = np.random.uniform(28, 33)  # Water bodies are cool

    # ── Coordinate grids ─────────────────────────────────────────────────────
    lat_extent = (rows * g["resolution_m"]) / 111320.0  # degrees
    lon_extent = (cols * g["resolution_m"]) / (111320.0 * np.cos(np.deg2rad(city["center_lat"])))
    lats = np.linspace(city["center_lat"] + lat_extent / 2,
                       city["center_lat"] - lat_extent / 2, rows)
    lons = np.linspace(city["center_lon"] - lon_extent / 2,
                       city["center_lon"] + lon_extent / 2, cols)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    print(f"  [OK] Data generated. LST range: {lst.min():.1f}°C – {lst.max():.1f}°C")
    print(f"  [OK] NDVI range: {ndvi.min():.2f} – {ndvi.max():.2f}")

    return {
        "lulc": lulc,
        "lst": lst,
        "ndvi": ndvi,
        "albedo": albedo,
        "svf": svf,
        "building_height": building_height,
        "impervious_fraction": impervious_fraction,
        "air_temp": air_temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "water_mask": water_mask,
        "rows": rows,
        "cols": cols,
    }


def grid_to_dataframe(data: dict) -> pd.DataFrame:
    """Flatten 2D grids into a feature DataFrame for ML training."""
    rows, cols = data["rows"], data["cols"]
    n = rows * cols

    df = pd.DataFrame({
        "lulc": data["lulc"].ravel(),
        "ndvi": data["ndvi"].ravel(),
        "albedo": data["albedo"].ravel(),
        "svf": data["svf"].ravel(),
        "building_height": data["building_height"].ravel(),
        "impervious_fraction": data["impervious_fraction"].ravel(),
        "air_temp": data["air_temp"].ravel(),
        "humidity": data["humidity"].ravel(),
        "wind_speed": data["wind_speed"].ravel(),
        "lat": data["lat_grid"].ravel(),
        "lon": data["lon_grid"].ravel(),
        "lst": data["lst"].ravel(),
        "is_water": data["water_mask"].ravel().astype(int),
    })

    # One-hot encode LULC
    lulc_dummies = pd.get_dummies(df["lulc"], prefix="lulc")
    df = pd.concat([df, lulc_dummies], axis=1)

    return df


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    data = generate_city_grid(cfg)
    df = grid_to_dataframe(data)
    print(df.describe())

