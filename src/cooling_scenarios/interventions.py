"""
interventions.py
----------------
Defines urban cooling interventions and their physical effects on
land surface features (NDVI, albedo, humidity, SVF).

Each intervention modifies the feature grid, and the ML model
then re-predicts LST to quantify the temperature reduction.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Intervention:
    """Represents a single type of urban cooling intervention."""
    name: str
    label: str
    ndvi_delta: float = 0.0
    albedo_delta: float = 0.0
    humidity_delta: float = 0.0
    impervious_delta: float = 0.0
    svf_delta: float = 0.0
    cost_per_cell: float = 10.0
    color: str = "#4CAF50"
    applicable_lulc: list = field(default_factory=lambda: list(range(8)))

    def apply(self, df_grid: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        """
        Apply intervention to selected cells in the feature grid.
        
        Parameters
        ----------
        df_grid : Full feature DataFrame
        mask    : Boolean array (flattened) indicating intervention cells
        
        Returns modified copy of df_grid.
        """
        df_mod = df_grid.copy()
        idx = np.where(mask)[0]

        # Apply deltas with physical bounds
        if self.ndvi_delta != 0 and "ndvi" in df_mod.columns:
            df_mod.loc[idx, "ndvi"] = np.clip(
                df_mod.loc[idx, "ndvi"] + self.ndvi_delta, -0.1, 0.85)

        if self.albedo_delta != 0 and "albedo" in df_mod.columns:
            df_mod.loc[idx, "albedo"] = np.clip(
                df_mod.loc[idx, "albedo"] + self.albedo_delta, 0.05, 0.80)

        if self.humidity_delta != 0 and "humidity" in df_mod.columns:
            df_mod.loc[idx, "humidity"] = np.clip(
                df_mod.loc[idx, "humidity"] + self.humidity_delta * 10, 20, 95)

        if self.impervious_delta != 0 and "impervious_fraction" in df_mod.columns:
            df_mod.loc[idx, "impervious_fraction"] = np.clip(
                df_mod.loc[idx, "impervious_fraction"] + self.impervious_delta, 0, 1)

        if self.svf_delta != 0 and "svf" in df_mod.columns:
            df_mod.loc[idx, "svf"] = np.clip(
                df_mod.loc[idx, "svf"] + self.svf_delta, 0.1, 1.0)

        return df_mod


# ── Predefined Intervention Library ──────────────────────────────────────────

INTERVENTIONS = {
    "urban_greening": Intervention(
        name="urban_greening",
        label="Urban Greening (Trees + Parks)",
        ndvi_delta=+0.30,
        albedo_delta=-0.05,
        humidity_delta=+0.15,
        impervious_delta=-0.20,
        cost_per_cell=8,
        color="#2d8a4e",
        applicable_lulc=[4, 5, 6, 7],  # Residential, commercial, barren
    ),

    "cool_roofs": Intervention(
        name="cool_roofs",
        label="Cool Roofs (High-Albedo Coating)",
        albedo_delta=+0.25,
        ndvi_delta=0.0,
        impervious_delta=0.0,
        cost_per_cell=5,
        color="#CFD8DC",
        applicable_lulc=[4, 5, 6],     # Residential and commercial buildings
    ),

    "green_roofs": Intervention(
        name="green_roofs",
        label="Green Roofs (Rooftop Vegetation)",
        ndvi_delta=+0.15,
        albedo_delta=+0.05,
        humidity_delta=+0.10,
        impervious_delta=-0.10,
        cost_per_cell=12,
        color="#6abf69",
        applicable_lulc=[4, 5, 6],
    ),

    "water_bodies": Intervention(
        name="water_bodies",
        label="Water Bodies / Blue Infrastructure",
        ndvi_delta=+0.05,
        albedo_delta=-0.10,
        humidity_delta=+0.25,
        impervious_delta=-0.30,
        cost_per_cell=20,
        color="#4fc3f7",
        applicable_lulc=[4, 5, 6, 7],
    ),

    "permeable_pavements": Intervention(
        name="permeable_pavements",
        label="Permeable Pavements",
        albedo_delta=+0.10,
        impervious_delta=-0.40,
        humidity_delta=+0.08,
        cost_per_cell=7,
        color="#BCAAA4",
        applicable_lulc=[4, 5, 6],
    ),

    "street_trees": Intervention(
        name="street_trees",
        label="Street Trees & Shading",
        ndvi_delta=+0.20,
        svf_delta=-0.10,
        humidity_delta=+0.12,
        cost_per_cell=6,
        color="#388E3C",
        applicable_lulc=[4, 5, 6],
    ),
}


def get_intervention(name: str) -> Intervention:
    """Retrieve an intervention by name."""
    if name not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention '{name}'. Available: {list(INTERVENTIONS.keys())}")
    return INTERVENTIONS[name]


def get_hotspot_application_mask(hotspot_class: np.ndarray,
                                 lulc: np.ndarray,
                                 intervention: Intervention,
                                 coverage_fraction: float = 0.5) -> np.ndarray:
    """
    Generate application mask: apply intervention to top hotspot cells
    that have applicable LULC classes.
    
    Returns flattened boolean mask.
    """
    rows, cols = hotspot_class.shape
    hot_flat = (hotspot_class >= 1).ravel()
    lulc_flat = lulc.ravel()

    # Applicable LULC mask
    lulc_ok = np.isin(lulc_flat, intervention.applicable_lulc)
    candidate_mask = hot_flat & lulc_ok

    # Limit to coverage_fraction of candidates
    candidate_indices = np.where(candidate_mask)[0]
    n_apply = max(1, int(len(candidate_indices) * coverage_fraction))
    selected = candidate_indices[:n_apply]  # Top hotspot cells first

    mask = np.zeros(rows * cols, dtype=bool)
    mask[selected] = True
    return mask

