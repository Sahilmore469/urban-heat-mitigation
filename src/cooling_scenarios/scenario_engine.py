"""
scenario_engine.py
------------------
Simulates urban cooling scenarios by applying interventions to the
feature grid and re-predicting LST with the trained ML model.

For each intervention:
  1. Apply feature deltas to hotspot cells (applicable LULC only)
  2. Re-predict LST using the PIML model
  3. Compute ΔT (temperature reduction), spatial extent, cost-effectiveness
  4. Return structured results for comparison
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.cooling_scenarios.interventions import (
    INTERVENTIONS, get_intervention, get_hotspot_application_mask
)


def run_scenario(
    intervention_name: str,
    df_baseline: pd.DataFrame,
    lst_baseline: np.ndarray,
    model,
    feature_cols: list,
    hotspot_class: np.ndarray,
    lulc_grid: np.ndarray,
    config: dict,
    coverage_fraction: float = 0.6,
) -> dict:
    """
    Run a single cooling scenario simulation.
    
    Parameters
    ----------
    intervention_name : name of intervention from INTERVENTIONS dict
    df_baseline       : baseline feature DataFrame (full grid)
    lst_baseline      : 2D baseline LST array
    model             : trained PIML/XGBoost model
    feature_cols      : feature column names for model input
    hotspot_class     : 2D Gi* hotspot classification array
    lulc_grid         : 2D LULC array
    config            : config dict
    coverage_fraction : fraction of hotspot cells to intervene on
    
    Returns
    -------
    dict with scenario results
    """
    intervention = get_intervention(intervention_name)
    rows, cols = lst_baseline.shape

    # Build application mask (apply only to hotspot + applicable LULC cells)
    mask = get_hotspot_application_mask(
        hotspot_class, lulc_grid, intervention, coverage_fraction
    )
    n_cells_intervened = int(mask.sum())

    if n_cells_intervened == 0:
        print(f"  [WARN] No applicable cells for {intervention.label}")
        return None

    # Apply intervention to feature grid
    df_modified = intervention.apply(df_baseline, mask)

    # Re-predict LST
    X_modified = df_modified[feature_cols].astype(float)
    lst_predicted_flat = model.predict(X_modified)
    lst_scenario = lst_predicted_flat.reshape(rows, cols)

    # Compute deltas
    delta_lst = lst_scenario - lst_baseline  # Negative = cooling
    delta_in_hotspots = delta_lst.ravel()[mask]

    mean_cooling = float(-delta_in_hotspots.mean())   # Positive = cooling
    max_cooling = float(-delta_in_hotspots.min())
    median_cooling = float(-np.median(delta_in_hotspots))

    # Cost estimation
    total_cost = n_cells_intervened * intervention.cost_per_cell
    cost_per_degree = total_cost / (mean_cooling + 1e-6)

    # Area covered (using grid resolution from config)
    res_m = config["grid"]["resolution_m"]
    area_km2 = n_cells_intervened * (res_m / 1000.0)**2

    result = {
        "intervention": intervention_name,
        "label": intervention.label,
        "color": intervention.color,
        "n_cells_intervened": n_cells_intervened,
        "area_covered_km2": round(area_km2, 2),
        "mean_cooling_C": round(mean_cooling, 3),
        "median_cooling_C": round(median_cooling, 3),
        "max_cooling_C": round(max_cooling, 3),
        "total_cost_units": round(total_cost, 1),
        "cost_per_degree_C": round(cost_per_degree, 1),
        "lst_scenario": lst_scenario,
        "delta_lst": delta_lst,
        "mask": mask.reshape(rows, cols),
    }

    print(f"  [OK] {intervention.label}")
    print(f"     Cells: {n_cells_intervened} | Area: {area_km2:.1f} km² | "
          f"Mean ΔT: -{mean_cooling:.2f}°C | Cost: {total_cost:.0f} units")

    return result


def run_all_scenarios(
    df_baseline: pd.DataFrame,
    lst_baseline: np.ndarray,
    model,
    feature_cols: list,
    hotspot_class: np.ndarray,
    lulc_grid: np.ndarray,
    config: dict,
) -> dict:
    """
    Run all available cooling scenarios and return comparative results.
    """
    print("\n  [TEMP] Running cooling scenario simulations...")
    results = {}

    for name in INTERVENTIONS.keys():
        result = run_scenario(
            intervention_name=name,
            df_baseline=df_baseline,
            lst_baseline=lst_baseline,
            model=model,
            feature_cols=feature_cols,
            hotspot_class=hotspot_class,
            lulc_grid=lulc_grid,
            config=config,
            coverage_fraction=0.65,
        )
        if result is not None:
            results[name] = result

    # Summary table
    summary = []
    for name, r in results.items():
        summary.append({
            "Intervention": r["label"],
            "Area (km²)": r["area_covered_km2"],
            "Mean ΔT (°C)": -r["mean_cooling_C"],
            "Max ΔT (°C)": -r["max_cooling_C"],
            "Cost (units)": r["total_cost_units"],
            "Cost/°C": r["cost_per_degree_C"],
        })

    summary_df = pd.DataFrame(summary).sort_values("Mean ΔT (°C)")

    print("\n  [CHART] Scenario Comparison:")
    print(summary_df.to_string(index=False))

    return {
        "scenarios": results,
        "summary_df": summary_df,
    }


def run_combined_scenario(
    scenario_names: list,
    df_baseline: pd.DataFrame,
    lst_baseline: np.ndarray,
    model,
    feature_cols: list,
    hotspot_class: np.ndarray,
    lulc_grid: np.ndarray,
    config: dict,
) -> dict:
    """
    Run a combined (multi-intervention) scenario by stacking interventions.
    """
    print(f"\n  [MIX] Combined scenario: {' + '.join(scenario_names)}")
    df_combined = df_baseline.copy()
    rows, cols = lst_baseline.shape

    all_masks = np.zeros(rows * cols, dtype=bool)

    for name in scenario_names:
        intervention = get_intervention(name)
        mask = get_hotspot_application_mask(
            hotspot_class, lulc_grid, intervention, coverage_fraction=0.5
        )
        df_combined = intervention.apply(df_combined, mask)
        all_masks = all_masks | mask

    X_combined = df_combined[feature_cols].astype(float)
    lst_combined_flat = model.predict(X_combined)
    lst_combined = lst_combined_flat.reshape(rows, cols)

    delta_combined = lst_combined - lst_baseline
    mean_cooling = float(-delta_combined.ravel()[all_masks].mean())

    total_cost = sum(
        int(all_masks.sum()) * get_intervention(n).cost_per_cell
        for n in scenario_names
    )

    print(f"     Combined Mean Cooling: -{mean_cooling:.2f}°C")
    print(f"     Combined Cost: {total_cost:.0f} units")

    return {
        "interventions": scenario_names,
        "lst_combined": lst_combined,
        "delta_combined": delta_combined,
        "combined_mask": all_masks.reshape(rows, cols),
        "mean_cooling_C": round(mean_cooling, 3),
        "total_cost_units": total_cost,
    }

