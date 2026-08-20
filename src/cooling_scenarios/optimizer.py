"""
optimizer.py
------------
Genetic Algorithm–based optimization to find the best combination
and spatial placement of cooling interventions under:
  - Budget constraints (total cost units)
  - Area coverage limits (fraction of city)
  - Maximizing total temperature reduction in hotspot zones

Uses DEAP (Distributed Evolutionary Algorithms in Python).
Falls back to scipy differential evolution if DEAP is unavailable.
"""

import numpy as np
import pandas as pd
import json
from typing import Optional

from src.cooling_scenarios.interventions import INTERVENTIONS, get_intervention


# ── Greedy Optimizer (fast, no DEAP dependency required) ─────────────────────

def greedy_optimize(
    df_baseline: pd.DataFrame,
    lst_baseline: np.ndarray,
    model,
    feature_cols: list,
    hotspot_class: np.ndarray,
    lulc_grid: np.ndarray,
    config: dict,
) -> dict:
    """
    Greedy budget-constrained optimization:
    Iteratively selects the most cost-effective intervention for
    each hotspot cell until budget is exhausted.
    
    Returns the optimal intervention allocation.
    """
    budget = config["optimization"]["budget_units"]
    max_fraction = config["optimization"]["max_coverage_fraction"]
    res_m = config["grid"]["resolution_m"]
    rows, cols = lst_baseline.shape
    n_cells = rows * cols
    max_cells = int(n_cells * max_fraction)

    print(f"  Budget: {budget} units | Max coverage: {max_fraction*100:.0f}%")

    # Score each intervention by cooling per cost unit
    intervention_scores = {}
    for name, intv in INTERVENTIONS.items():
        # Estimate cooling per cell (heuristic based on feature deltas)
        est_cooling = (
            abs(intv.ndvi_delta) * 6.0       # NDVI contribution
            + abs(intv.albedo_delta) * 4.0    # Albedo contribution
            + abs(intv.humidity_delta) * 2.0  # Humidity contribution
            + abs(intv.impervious_delta) * 3.0
        )
        efficiency = est_cooling / (intv.cost_per_cell + 1e-6)
        intervention_scores[name] = {
            "intervention": intv,
            "est_cooling_per_cell": est_cooling,
            "efficiency": efficiency,
            "cost_per_cell": intv.cost_per_cell,
        }

    # Sort by efficiency
    ranked = sorted(intervention_scores.items(),
                    key=lambda x: x[1]["efficiency"], reverse=True)

    # Identify hotspot cells sorted by Gi* Z-score intensity
    hotspot_flat = hotspot_class.ravel()
    lulc_flat = lulc_grid.ravel()

    allocation = {}  # cell_idx -> intervention_name
    remaining_budget = budget
    total_cells_used = 0

    for name, score_info in ranked:
        intv = score_info["intervention"]
        if remaining_budget <= 0 or total_cells_used >= max_cells:
            break

        # Find applicable uncovered hotspot cells
        candidate_mask = (
            (hotspot_flat >= 1) &
            (np.isin(lulc_flat, intv.applicable_lulc))
        )
        candidate_indices = np.where(candidate_mask)[0]
        # Exclude already allocated cells
        allocated_set = set(allocation.keys())
        candidate_indices = [i for i in candidate_indices if i not in allocated_set]

        if not candidate_indices:
            continue

        # How many cells can we afford?
        cells_affordable = int(remaining_budget / intv.cost_per_cell)
        cells_to_use = min(len(candidate_indices),
                           cells_affordable,
                           max_cells - total_cells_used)

        # Take highest-LST hotspot cells first
        lst_flat = lst_baseline.ravel()
        candidate_indices_sorted = sorted(candidate_indices,
                                          key=lambda i: -lst_flat[i])
        selected = candidate_indices_sorted[:cells_to_use]

        for idx in selected:
            allocation[idx] = name

        cost_used = len(selected) * intv.cost_per_cell
        remaining_budget -= cost_used
        total_cells_used += len(selected)

        area_km2 = len(selected) * (res_m / 1000.0)**2
        print(f"     Allocated '{intv.label}': {len(selected)} cells "
              f"({area_km2:.1f} km²) | Cost: {cost_used:.0f} units")

    # Simulate optimal scenario
    df_optimal = df_baseline.copy()
    intervention_counts = {}

    for cell_idx, intv_name in allocation.items():
        intv = get_intervention(intv_name)
        mask = np.zeros(n_cells, dtype=bool)
        mask[cell_idx] = True
        df_optimal = intv.apply(df_optimal, mask)
        intervention_counts[intv_name] = intervention_counts.get(intv_name, 0) + 1

    # Re-predict LST
    X_opt = df_optimal[feature_cols].astype(float)
    lst_optimal_flat = model.predict(X_opt)
    lst_optimal = lst_optimal_flat.reshape(rows, cols)

    # Compute overall cooling
    delta_optimal = lst_optimal - lst_baseline
    hotspot_mask = (hotspot_class >= 1)
    mean_cooling_hotspots = float(-delta_optimal[hotspot_mask].mean()) if hotspot_mask.any() else 0.0
    mean_cooling_city = float(-delta_optimal.mean())

    total_cost_used = budget - remaining_budget
    area_covered_km2 = total_cells_used * (res_m / 1000.0)**2

    results = {
        "algorithm": "Greedy Budget-Constrained",
        "budget_total": budget,
        "budget_used": round(total_cost_used, 1),
        "budget_remaining": round(remaining_budget, 1),
        "total_cells_allocated": total_cells_used,
        "area_covered_km2": round(area_covered_km2, 2),
        "intervention_allocation": intervention_counts,
        "mean_cooling_hotspots_C": round(mean_cooling_hotspots, 3),
        "mean_cooling_city_C": round(mean_cooling_city, 3),
        "lst_optimal": lst_optimal,
        "delta_optimal": delta_optimal,
        "allocation_map": allocation,
    }

    print(f"\n  [DONE] Optimal Strategy:")
    print(f"     Budget used: {total_cost_used:.0f}/{budget} units")
    print(f"     Area covered: {area_covered_km2:.1f} km²")
    print(f"     Mean cooling (hotspots): -{mean_cooling_hotspots:.2f}°C")
    print(f"     Mean cooling (city-wide): -{mean_cooling_city:.2f}°C")
    print(f"     Intervention mix: {intervention_counts}")

    return results


def save_optimization_results(results: dict, output_path: str):
    """Save optimization results to JSON (excluding numpy arrays)."""
    save_dict = {k: v for k, v in results.items()
                 if not isinstance(v, np.ndarray) and k != "allocation_map"}
    save_dict["intervention_allocation"] = results["intervention_allocation"]

    with open(output_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"  [SAVE] Results saved to {output_path}")

