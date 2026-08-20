"""
run_extended.py  --  Stages 9-11
Extended pipeline: LSTM Temporal + Interactive Map + PDF Report
Usage: python run_extended.py
"""

import os, sys, time, warnings, json
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"

MAPS_DIR    = "outputs/maps"
FIGS_DIR    = "outputs/figures"
REPORTS_DIR = "outputs/reports"
OUTPUT_DIR  = "outputs"


def pstage(n, title):
    print(f"\n{'='*78}\n  STAGE {n}: {title.upper()}\n{'='*78}")


def main():
    t0 = time.time()
    print("\n================================================================================")
    print("  URBAN HEAT MITIGATION -- EXTENDED PIPELINE")
    print("  LSTM Temporal + Interactive Map + PDF Report")
    print("================================================================================")

    # -- Load base pipeline ---------------------------------------------------
    from src.data_pipeline.synthetic_data import (
        load_config, generate_city_grid, grid_to_dataframe)
    from src.heat_analysis.heat_stress_index import (
        utci_approximation, classify_heat_stress)
    from src.heat_analysis.hotspot_detector import detect_hotspots
    from src.ml_models.piml_model import (
        train_piml_pipeline, predict_full_grid)
    from src.cooling_scenarios.optimizer import greedy_optimize

    config     = load_config("config.yaml")
    print(f"  City: {config['city']['name']}")
    print("  Loading base pipeline data...")
    data       = generate_city_grid(config)
    df         = grid_to_dataframe(data)
    utci       = utci_approximation(
        data["air_temp"], data["humidity"], data["lst"], data["wind_speed"])
    hotspot_r  = detect_hotspots(data["lst"], utci, config)
    ml_results = train_piml_pipeline(df, config)
    model      = ml_results["model"]
    feat_cols  = ml_results["feature_cols"]
    lst_pred   = predict_full_grid(
        model, df, feat_cols).reshape(data["rows"], data["cols"])
    opt_r      = greedy_optimize(
        df, lst_pred, model, feat_cols,
        hotspot_r["hotspot_class"], data["lulc"], config)

    # -- Stage 9: LSTM Temporal Model -----------------------------------------
    pstage(9, "LSTM Temporal Heat Dynamics")
    from src.ml_models.lstm_temporal import train_lstm_temporal, plot_temporal_results
    lstm_results = train_lstm_temporal(config)
    plot_temporal_results(lstm_results, FIGS_DIR)
    lstm_path = os.path.join(REPORTS_DIR, "lstm_metrics.json")
    with open(lstm_path, "w") as f:
        json.dump(lstm_results["metrics"], f, indent=2)
    print(f"  [SAVE] LSTM metrics: {lstm_path}")

    # -- Stage 10: Folium Interactive Map -------------------------------------
    pstage(10, "Folium Interactive HTML Map")
    from src.visualization.interactive_map import make_interactive_heat_map
    stress_class = classify_heat_stress(utci)
    html_path = make_interactive_heat_map(
        lst           = data["lst"],
        hotspot_class = hotspot_r["hotspot_class"],
        stress_class  = stress_class,
        lat_grid      = data["lat_grid"],
        lon_grid      = data["lon_grid"],
        opt_allocation= opt_r.get("allocation_map", {}),
        lulc          = data["lulc"],
        config        = config,
        output_dir    = OUTPUT_DIR,
    )

    # -- Stage 11: PDF Report --------------------------------------------------
    pstage(11, "Professional PDF Report")
    from src.visualization.pdf_report import generate_pdf_report

    opt_for_pdf = {
        "budget_total":            config["optimization"]["budget_units"],
        "budget_used":             opt_r.get("budget_used", 0),
        "area_covered_km2":        opt_r.get("area_covered_km2", 0),
        "total_cells_allocated":   opt_r.get("total_cells_allocated", 0),
        "mean_cooling_hotspots_C": opt_r.get("mean_cooling_hotspots_C", 0),
        "mean_cooling_city_C":     opt_r.get("mean_cooling_city_C", 0),
        "intervention_allocation": opt_r.get("intervention_allocation", {}),
    }
    pdf_path = generate_pdf_report(
        maps_dir     = MAPS_DIR,
        figs_dir     = FIGS_DIR,
        reports_dir  = REPORTS_DIR,
        output_dir   = OUTPUT_DIR,
        config       = config,
        hotspot_stats= hotspot_r["hotspot_stats"],
        metrics      = ml_results["metrics"],
        lstm_metrics = lstm_results["metrics"],
        opt_results  = opt_for_pdf,
    )

    # -- Done ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*78}")
    print(f"  [DONE] EXTENDED PIPELINE COMPLETE  |  Runtime: {elapsed:.1f}s")
    print(f"{'='*78}")
    print(f"\n  New outputs:")
    print(f"    LSTM figure : {FIGS_DIR}/lstm_temporal_analysis.png")
    print(f"    LSTM metrics: {lstm_path}")
    print(f"    Interactive : {html_path}")
    print(f"    PDF report  : {pdf_path}")
    print(f"\n  Open interactive map:")
    print(f"    start outputs\\interactive_heat_map.html")
    print(f"  Open PDF report:")
    print(f"    start outputs\\urban_heat_report.pdf")


if __name__ == "__main__":
    main()
