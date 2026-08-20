"""
hotspot_detector.py
-------------------
Identifies statistically significant urban heat stress hotspots using:
  1. Getis-Ord Gi* spatial statistic (spatial clustering of high LST values)
  2. Local Moran's I (LISA) for spatial outlier detection
  3. Kernel Density Estimation (KDE) for smooth hotspot intensity surfaces

These methods are standard in spatial epidemiology and urban climatology.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import norm


def getis_ord_gi_star(arr: np.ndarray, window_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute simplified Getis-Ord Gi* statistic using a moving window.
    
    Gi* measures the degree to which a location and its neighbors have
    high (or low) values relative to the global mean.
    
    Parameters
    ----------
    arr : 2D array of the variable (e.g., LST or UTCI)
    window_size : neighborhood window size (must be odd)
    
    Returns
    -------
    gi_star : Z-score surface (positive = hot spot, negative = cold spot)
    p_value  : Two-tailed p-values for each cell
    """
    rows, cols = arr.shape
    n = rows * cols
    x_bar = arr.mean()
    s = arr.std()

    # Local sum and count using uniform filter
    kernel = np.ones((window_size, window_size))
    w_sum = ndimage.uniform_filter(arr, size=window_size, mode='reflect') * window_size**2
    w_count = window_size**2  # scalar (all weights = 1)

    # Gi* formula
    numerator = w_sum - x_bar * w_count
    denominator = s * np.sqrt((n * w_count - w_count**2) / (n - 1))

    gi_star = numerator / (denominator + 1e-10)

    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(np.abs(gi_star)))

    return gi_star, p_value


def classify_hotspots(gi_star: np.ndarray, p_value: np.ndarray,
                       significance: float = 0.05) -> np.ndarray:
    """
    Classify spatial clusters into hotspot/coldspot categories.
    
    Returns integer array:
        -2 = Significant Cold Spot (p < 0.01)
        -1 = Cold Spot (p < 0.05)
         0 = Not Significant
        +1 = Hot Spot (p < 0.05)
        +2 = Significant Hot Spot (p < 0.01)
    """
    classification = np.zeros_like(gi_star, dtype=int)

    # Hot spots
    classification[(gi_star > 0) & (p_value < 0.05) & (p_value >= 0.01)] = 1
    classification[(gi_star > 0) & (p_value < 0.01)] = 2

    # Cold spots
    classification[(gi_star < 0) & (p_value < 0.05) & (p_value >= 0.01)] = -1
    classification[(gi_star < 0) & (p_value < 0.01)] = -2

    return classification


def local_morans_i(arr: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Simplified Local Moran's I spatial autocorrelation.
    Identifies spatial outliers (high-low or low-high anomalies).
    """
    x_bar = arr.mean()
    s2 = arr.var()
    z = arr - x_bar

    # Spatial lag (mean of neighbors)
    z_lag = ndimage.uniform_filter(z, size=window_size, mode='reflect')

    local_i = z * z_lag / (s2 + 1e-10)
    return local_i


def kde_hotspot_surface(lst: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """
    Kernel Density Estimation of heat intensity surface.
    Used to create smooth hotspot probability maps.
    """
    # Normalize LST to [0,1] and apply Gaussian KDE
    lst_norm = (lst - lst.min()) / (lst.max() - lst.min() + 1e-10)
    kde_surface = ndimage.gaussian_filter(lst_norm, sigma=sigma)
    # Normalize to [0, 1] probability
    kde_surface = (kde_surface - kde_surface.min()) / (kde_surface.max() - kde_surface.min() + 1e-10)
    return kde_surface


def detect_hotspots(lst: np.ndarray, utci: np.ndarray,
                    config: dict) -> dict:
    """
    Run full hotspot detection pipeline.
    
    Returns
    -------
    dict with:
        gi_star_lst    : Gi* Z-scores for LST
        gi_star_utci   : Gi* Z-scores for UTCI
        p_value_lst    : p-values for LST hotspots
        hotspot_class  : Classified hotspot map (-2 to +2)
        kde_surface    : Smooth KDE hotspot intensity
        local_moran    : Local Moran's I values
        hotspot_stats  : Summary statistics dict
    """
    sig = config["output"]["hotspot_significance"]

    print("  Running Getis-Ord Gi* spatial statistics...")
    gi_lst, p_lst = getis_ord_gi_star(lst, window_size=7)
    gi_utci, p_utci = getis_ord_gi_star(utci, window_size=7)

    print("  Classifying hotspot zones...")
    hotspot_class = classify_hotspots(gi_lst, p_lst, significance=sig)

    print("  Computing Local Moran's I...")
    local_moran = local_morans_i(lst, window_size=5)

    print("  Computing KDE hotspot surface...")
    kde_surf = kde_hotspot_surface(lst, sigma=10.0)

    total_cells = lst.size
    n_extreme_hot = np.sum(hotspot_class == 2)
    n_hot = np.sum(hotspot_class >= 1)
    n_cold = np.sum(hotspot_class <= -1)
    hotspot_fraction = n_hot / total_cells * 100

    # Mean LST in hotspot zones
    hot_mask = hotspot_class >= 1
    mean_lst_hotspot = float(lst[hot_mask].mean()) if hot_mask.any() else float(lst.mean())
    mean_lst_overall = float(lst.mean())
    lst_excess = mean_lst_hotspot - mean_lst_overall

    stats = {
        "total_cells": int(total_cells),
        "n_extreme_hotspots": int(n_extreme_hot),
        "n_hotspot_cells": int(n_hot),
        "n_coldspot_cells": int(n_cold),
        "hotspot_fraction_pct": round(hotspot_fraction, 2),
        "mean_lst_hotspot_C": round(mean_lst_hotspot, 2),
        "mean_lst_overall_C": round(mean_lst_overall, 2),
        "lst_excess_C": round(lst_excess, 2),
        "max_gi_star": round(float(gi_lst.max()), 3),
    }

    print(f"  [OK] Hotspots: {n_hot} cells ({hotspot_fraction:.1f}% of city)")
    print(f"  [OK] Mean LST in hotspots: {mean_lst_hotspot:.1f}°C (excess: +{lst_excess:.1f}°C)")

    return {
        "gi_star_lst": gi_lst,
        "gi_star_utci": gi_utci,
        "p_value_lst": p_lst,
        "hotspot_class": hotspot_class,
        "kde_surface": kde_surf,
        "local_moran": local_moran,
        "hotspot_stats": stats,
    }

