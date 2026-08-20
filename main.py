"""
main.py
-------
End-to-end pipeline for Urban Heat Mitigation Analysis.

Stages:
  1. Data Generation (synthetic physics-grounded city grid)
  2. Heat Stress Index Computation (UTCI / WBGT)
  3. Hotspot Detection (Getis-Ord Gi*)
  4. Physics-Informed ML Model Training & Validation
  5. Driver Analysis (SHAP)
  6. Cooling Scenario Simulation
  7. Optimization (Budget-constrained greedy)
  8. Visualization & Report Generation

Usage:
  python main.py
  python main.py --config config.yaml --output-dir outputs/
"""

import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

# Fix Windows console encoding for UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    banner = """
================================================================================
  URBAN HEAT MITIGATION AI/ML SYSTEM
  Geospatial Physics-Informed Analysis & Cooling Optimization
================================================================================
"""
    print(banner)


def print_stage(n: int, title: str):
    print(f"\n{'='*78}")
    print(f"  STAGE {n}: {title.upper()}")
    print(f"{'='*78}")


def run_pipeline(config_path: str = "config.yaml",
                 output_dir: str = "outputs") -> dict:
    """Execute the full urban heat mitigation analysis pipeline."""

    t_start = time.time()
    print_banner()

    # ── Load configuration ────────────────────────────────────────────────────
    from src.data_pipeline.synthetic_data import load_config, generate_city_grid, grid_to_dataframe
    config = load_config(config_path)
    city_name = config["city"]["name"]
    print(f"  [LOC] City: {city_name} ({config['city']['center_lat']}°N, {config['city']['center_lon']}°E)")
    print(f"  [DATE]  Analysis time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    maps_dir = os.path.join(output_dir, "maps")
    figs_dir = os.path.join(output_dir, "figures")
    reports_dir = os.path.join(output_dir, "reports")
    for d in [maps_dir, figs_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    print_stage(1, "Data Generation")
    print(f"  Generating physics-grounded synthetic city grid...")
    data = generate_city_grid(config)
    df = grid_to_dataframe(data)
    print(f"  [OK] Grid: {data['rows']}×{data['cols']} = {data['rows']*data['cols']:,} cells")
    df.to_csv(os.path.join(output_dir, "reports", "feature_grid.csv"), index=False)

    # ─────────────────────────────────────────────────────────────────────────
    print_stage(2, "Heat Stress Index Computation")
    from src.heat_analysis.heat_stress_index import (
        utci_approximation, wbgt_outdoor, classify_heat_stress,
        HEAT_STRESS_LABELS
    )

    utci = utci_approximation(
        air_temp=data["air_temp"],
        humidity=data["humidity"],
        lst=data["lst"],
        wind_speed=data["wind_speed"],
    )
    wbgt = wbgt_outdoor(
        air_temp=data["air_temp"],
        humidity=data["humidity"],
        lst=data["lst"],
        wind_speed=data["wind_speed"],
    )
    stress_class = classify_heat_stress(utci)

    print(f"  UTCI range: {utci.min():.1f}°C – {utci.max():.1f}°C")
    print(f"  WBGT range: {wbgt.min():.1f}°C – {wbgt.max():.1f}°C")
    print("\n  Heat Stress Distribution:")
    for level in range(6):
        frac = (stress_class == level).sum() / stress_class.size * 100
        bar = "█" * int(frac / 2)
        print(f"    Level {level} [{HEAT_STRESS_LABELS[level]:25s}]: {frac:5.1f}%  {bar}")

    # ─────────────────────────────────────────────────────────────────────────
    print_stage(3, "Urban Heat Hotspot Detection")
    from src.heat_analysis.hotspot_detector import detect_hotspots
    hotspot_results = detect_hotspots(data["lst"], utci, config)
    hs = hotspot_results["hotspot_stats"]
    print(f"\n  Hotspot Summary:")
    for k, v in hs.items():
        print(f"    {k}: {v}")

    # ─────────────────────────────────────────────────────────────────────────
    print_stage(4, "Physics-Informed ML Model Training & Validation")
    from src.ml_models.piml_model import train_piml_pipeline, predict_full_grid
    ml_results = train_piml_pipeline(df, config)
    model = ml_results["model"]
    feature_cols = ml_results["feature_cols"]
    metrics = ml_results["metrics"]

    # Predict full-grid LST for mapping
    lst_predicted_flat = predict_full_grid(model, df, feature_cols)
    lst_predicted = lst_predicted_flat.reshape(data["rows"], data["cols"])

    # ─────────────────────────────────────────────────────────────────────────
    print_stage(5, "Driver Analysis (SHAP)")
    from src.heat_analysis.driver_analysis import analyze_drivers
    driver_results = analyze_drivers(
        model=model,
        X_train=ml_results["X_train"],
        feature_cols=feature_cols,
        n_shap_samples=3000,
    )
    driver_results["importance_df"].to_csv(
        os.path.join(reports_dir, "driver_importance.csv"), index=False)
    driver_results["grouped_df"].to_csv(
        os.path.join(reports_dir, "driver_categories.csv"), index=False)

    # ─────────────────────────────────────────────────────────────────────────
    print_stage(6, "Cooling Scenario Simulation")
    from src.cooling_scenarios.scenario_engine import run_all_scenarios, run_combined_scenario

    scenario_results = run_all_scenarios(
        df_baseline=df,
        lst_baseline=lst_predicted,
        model=model,
        feature_cols=feature_cols,
        hotspot_class=hotspot_results["hotspot_class"],
        lulc_grid=data["lulc"],
        config=config,
    )
    scenario_results["summary_df"].to_csv(
        os.path.join(reports_dir, "scenario_summary.csv"), index=False)

    # Best combined scenario
    combined = run_combined_scenario(
        scenario_names=["urban_greening", "cool_roofs"],
        df_baseline=df,
        lst_baseline=lst_predicted,
        model=model,
        feature_cols=feature_cols,
        hotspot_class=hotspot_results["hotspot_class"],
        lulc_grid=data["lulc"],
        config=config,
    )

    # ─────────────────────────────────────────────────────────────────────────
    print_stage(7, "Optimization (Budget-Constrained)")
    from src.cooling_scenarios.optimizer import greedy_optimize, save_optimization_results
    opt_results = greedy_optimize(
        df_baseline=df,
        lst_baseline=lst_predicted,
        model=model,
        feature_cols=feature_cols,
        hotspot_class=hotspot_results["hotspot_class"],
        lulc_grid=data["lulc"],
        config=config,
    )
    save_optimization_results(opt_results, os.path.join(reports_dir, "optimization_results.json"))

    # ─────────────────────────────────────────────────────────────────────────
    print_stage(8, "Visualization & Report Generation")
    from src.visualization.heat_maps import (
        plot_lulc_map, plot_lst_map, plot_heat_stress_map,
        plot_hotspot_map, plot_optimal_strategy_map
    )
    from src.visualization.driver_plots import (
        plot_shap_summary, plot_driver_importance_bar,
        plot_driver_category_pie, plot_scenario_comparison,
        plot_model_validation
    )

    print("\n  Generating maps...")
    plot_lulc_map(data["lulc"], config, maps_dir)
    plot_lst_map(data["lst"], config, maps_dir, title_suffix=" (Simulated Landsat)",
                 filename="lst_baseline.png")
    plot_lst_map(lst_predicted, config, maps_dir, title_suffix=" (ML Predicted)",
                 filename="lst_predicted.png")
    plot_heat_stress_map(stress_class, config, maps_dir)
    plot_hotspot_map(
        hotspot_results["hotspot_class"],
        hotspot_results["gi_star_lst"],
        config, maps_dir
    )
    plot_optimal_strategy_map(
        lst_baseline=lst_predicted,
        lst_optimal=opt_results["lst_optimal"],
        delta_lst=opt_results["delta_optimal"],
        allocation_map=opt_results["allocation_map"],
        config=config,
        output_dir=maps_dir,
    )

    print("\n  Generating analysis figures...")
    plot_model_validation(
        y_test=ml_results["y_test"].values,
        y_pred=ml_results["y_pred_test"],
        metrics=metrics,
        output_dir=figs_dir,
    )
    plot_shap_summary(
        shap_values=driver_results["shap_values"],
        X_sample=driver_results["X_sample"],
        feature_names=feature_cols,
        output_dir=figs_dir,
    )
    plot_driver_importance_bar(driver_results["importance_df"], figs_dir)
    plot_driver_category_pie(driver_results["grouped_df"], figs_dir)
    plot_scenario_comparison(scenario_results["scenarios"], figs_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Final Summary Report
    t_elapsed = time.time() - t_start
    summary_report = {
        "city": city_name,
        "grid_size": f"{data['rows']}×{data['cols']}",
        "total_cells": data["rows"] * data["cols"],
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(t_elapsed, 1),

        "heat_stress": {
            "utci_mean_C": round(float(utci.mean()), 2),
            "utci_max_C": round(float(utci.max()), 2),
            "extreme_stress_pct": round(float((stress_class >= 4).mean() * 100), 2),
            "wbgt_mean_C": round(float(wbgt.mean()), 2),
        },

        "hotspots": hotspot_results["hotspot_stats"],

        "ml_model": metrics,

        "top_3_drivers": driver_results["importance_df"].head(3)[["label", "pct_contribution"]].to_dict("records"),

        "best_single_intervention": max(
            scenario_results["scenarios"].values(),
            key=lambda r: r["mean_cooling_C"]
        )["label"] if scenario_results["scenarios"] else "N/A",

        "optimal_strategy": {
            "budget_used": opt_results["budget_used"],
            "area_covered_km2": opt_results["area_covered_km2"],
            "mean_cooling_hotspots_C": opt_results["mean_cooling_hotspots_C"],
            "mean_cooling_city_C": opt_results["mean_cooling_city_C"],
            "intervention_mix": opt_results["intervention_allocation"],
        }
    }

    report_path = os.path.join(reports_dir, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n{'='*78}")
    print(f"  [DONE]  PIPELINE COMPLETE  |  Runtime: {t_elapsed:.1f}s")
    print(f"{'='*78}")
    print(f"\n  [CHART] Key Results:")
    print(f"     City: {city_name}")
    print(f"     Mean UTCI: {summary_report['heat_stress']['utci_mean_C']}°C")
    print(f"     Extreme Heat Stress: {summary_report['heat_stress']['extreme_stress_pct']}% of city")
    print(f"     Hotspot cells: {hs['n_hotspot_cells']:,} ({hs['hotspot_fraction_pct']}%)")
    print(f"     ML Model R²: {metrics['r2']}")
    print(f"     Optimal cooling (hotspots): -{opt_results['mean_cooling_hotspots_C']}°C")
    print(f"\n  [DIR] Outputs saved to: {os.path.abspath(output_dir)}/")
    print(f"     Maps    : {maps_dir}/")
    print(f"     Figures : {figs_dir}/")
    print(f"     Reports : {reports_dir}/")
    print(f"     Report  : {report_path}")

    return summary_report


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Urban Heat Mitigation AI/ML Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    # Change to script directory so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    results = run_pipeline(
        config_path=args.config,
        output_dir=args.output_dir,
    )
