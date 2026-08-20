"""
pdf_report.py
-------------
Generates a professional PDF report using matplotlib multi-page PDF backend.

Report sections:
  1. Cover page — city summary, key KPIs
  2. Heat stress maps (LULC, LST, UTCI)
  3. Hotspot analysis (Gi* map + stats table)
  4. ML model validation
  5. Driver analysis (SHAP bar + category pie)
  6. Cooling scenario comparison
  7. Optimal strategy map + recommendation table
  8. LSTM temporal heat dynamics
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import os
import json
import time


# ── Palette ───────────────────────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_CARD   = "#161b22"
BLUE      = "#1565C0"
RED       = "#C62828"
GREEN     = "#2E7D32"
ORANGE    = "#E65100"
WHITE     = "#FFFFFF"
LIGHT     = "#E3F2FD"


def _page_header(fig, title: str, subtitle: str = "", page_num: int = 0,
                 total_pages: int = 0):
    """Add a consistent header bar to each page."""
    fig.patch.set_facecolor(BG_DARK)
    # Header bar
    ax_hdr = fig.add_axes([0, 0.93, 1, 0.07])
    ax_hdr.set_facecolor(BLUE)
    ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    ax_hdr.text(0.02, 0.55, title, color=WHITE,
                fontsize=14, fontweight="bold", va="center")
    ax_hdr.text(0.02, 0.15, subtitle, color=LIGHT,
                fontsize=9, va="center")
    if total_pages > 0:
        ax_hdr.text(0.98, 0.5, f"Page {page_num}/{total_pages}",
                    color=LIGHT, fontsize=9, ha="right", va="center")


def _embed_image(ax, img_path: str, title: str = ""):
    """Embed a saved PNG into a matplotlib axes."""
    ax.set_facecolor(BG_CARD)
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img, aspect="auto")
    else:
        ax.text(0.5, 0.5, f"[{title}]", ha="center", va="center",
                color="gray", fontsize=10, transform=ax.transAxes)
    ax.axis("off")
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=5)


def _kpi_box(ax, value: str, label: str, color: str = BLUE):
    ax.set_facecolor(color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.65, value, ha="center", va="center", color=WHITE,
            fontsize=18, fontweight="bold")
    ax.text(0.5, 0.25, label, ha="center", va="center", color=LIGHT,
            fontsize=9)


def generate_pdf_report(
    maps_dir: str,
    figs_dir: str,
    reports_dir: str,
    output_dir: str,
    config: dict,
    hotspot_stats: dict,
    metrics: dict,
    lstm_metrics: dict,
    opt_results: dict,
) -> str:
    """Generate professional multi-page PDF report."""

    city = config["city"]["name"]
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    out_path = os.path.join(output_dir, "urban_heat_report.pdf")

    # Load report JSON
    report_json = {}
    rp = os.path.join(reports_dir, "final_report.json")
    if os.path.exists(rp):
        with open(rp) as f:
            report_json = json.load(f)

    scenario_df = pd.DataFrame()
    sp = os.path.join(reports_dir, "scenario_summary.csv")
    if os.path.exists(sp):
        scenario_df = pd.read_csv(sp)

    driver_df = pd.DataFrame()
    dp = os.path.join(reports_dir, "driver_importance.csv")
    if os.path.exists(dp):
        driver_df = pd.read_csv(dp)

    with PdfPages(out_path) as pdf:
        total_pages = 8

        # ── Page 1: Cover ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.patch.set_facecolor(BG_DARK)

        # Title block
        ax_title = fig.add_axes([0.05, 0.55, 0.90, 0.40])
        ax_title.set_facecolor(BLUE)
        ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis("off")
        ax_title.text(0.5, 0.75, "URBAN HEAT MITIGATION", ha="center", color=WHITE,
                      fontsize=26, fontweight="bold")
        ax_title.text(0.5, 0.52, "AI / ML ANALYSIS SYSTEM", ha="center", color=LIGHT,
                      fontsize=20)
        ax_title.text(0.5, 0.28, "Geospatial Physics-Informed Analysis & Cooling Optimization",
                      ha="center", color=LIGHT, fontsize=13)
        ax_title.text(0.5, 0.10, f"City: {city}   |   {timestamp}",
                      ha="center", color=LIGHT, fontsize=11)

        # KPI row
        kpis = [
            (f"{hotspot_stats.get('hotspot_fraction_pct', 0):.1f}%", "City Hotspot Area", RED),
            (f"+{hotspot_stats.get('lst_excess_C', 0):.1f}°C", "Urban Heat Excess", ORANGE),
            (f"{metrics.get('r2', 0):.4f}", "PIML Model R²", GREEN),
            (f"{opt_results.get('mean_cooling_hotspots_C', 0):.2f}°C", "Optimal Cooling", BLUE),
        ]
        for i, (val, lbl, col) in enumerate(kpis):
            ax_k = fig.add_axes([0.04 + i * 0.235, 0.38, 0.21, 0.14])
            _kpi_box(ax_k, val, lbl, col)

        # Footer info
        ax_foot = fig.add_axes([0.05, 0.05, 0.90, 0.28])
        ax_foot.set_facecolor(BG_CARD); ax_foot.axis("off")
        info = [
            ("Framework", "Physics-Informed XGBoost + LSTM + SHAP + Getis-Ord Gi*"),
            ("Data Sources", "Synthetic Landsat LST proxy, ERA5 atmospheric, OSM morphology"),
            ("Grid", f"{config['grid']['rows']}×{config['grid']['cols']} cells @ {config['grid']['resolution_m']}m resolution"),
            ("Interventions", "Urban Greening, Cool Roofs, Green Roofs, Water Bodies, Permeable Pavements, Street Trees"),
            ("Optimizer", f"Greedy budget-constrained, Budget={opt_results.get('budget_total',50000):,} units"),
        ]
        for j, (key, val) in enumerate(info):
            ax_foot.text(0.02, 0.85 - j * 0.17, f"{key}:", color="#90CAF9",
                         fontsize=10, fontweight="bold", transform=ax_foot.transAxes)
            ax_foot.text(0.22, 0.85 - j * 0.17, val, color=WHITE,
                         fontsize=10, transform=ax_foot.transAxes)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Heat Stress Maps ───────────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        _page_header(fig, "Urban Heat Stress Maps", f"City: {city}", 2, total_pages)
        fig.patch.set_facecolor(BG_DARK)

        axes_imgs = [
            fig.add_axes([0.03, 0.10, 0.30, 0.80]),
            fig.add_axes([0.35, 0.10, 0.30, 0.80]),
            fig.add_axes([0.67, 0.10, 0.30, 0.80]),
        ]
        _embed_image(axes_imgs[0], os.path.join(maps_dir, "lulc_map.png"),         "Land Use / Land Cover")
        _embed_image(axes_imgs[1], os.path.join(maps_dir, "lst_baseline.png"),      "Land Surface Temperature")
        _embed_image(axes_imgs[2], os.path.join(maps_dir, "heat_stress_map.png"),   "UTCI Heat Stress (6 Levels)")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 3: Hotspot Detection ──────────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        _page_header(fig, "Urban Heat Hotspot Detection — Getis-Ord Gi*",
                     f"City: {city}", 3, total_pages)
        fig.patch.set_facecolor(BG_DARK)

        ax_map = fig.add_axes([0.03, 0.10, 0.55, 0.80])
        _embed_image(ax_map, os.path.join(maps_dir, "hotspot_map.png"), "Gi* Classification + Z-Score Surface")

        ax_tbl = fig.add_axes([0.60, 0.10, 0.38, 0.80])
        ax_tbl.set_facecolor(BG_CARD); ax_tbl.axis("off")
        ax_tbl.text(0.5, 0.96, "Hotspot Statistics", ha="center", color=WHITE,
                    fontsize=12, fontweight="bold", transform=ax_tbl.transAxes)
        stats_items = [
            ("Total Cells", f"{hotspot_stats.get('total_cells', 0):,}"),
            ("Hotspot Cells", f"{hotspot_stats.get('n_hotspot_cells', 0):,}"),
            ("Hotspot Fraction", f"{hotspot_stats.get('hotspot_fraction_pct', 0):.2f}%"),
            ("Extreme Hotspots", f"{hotspot_stats.get('n_extreme_hotspots', 0):,}"),
            ("Cold Spot Cells", f"{hotspot_stats.get('n_coldspot_cells', 0):,}"),
            ("Mean LST (Hotspot)", f"{hotspot_stats.get('mean_lst_hotspot_C', 0):.2f}°C"),
            ("Mean LST (City)", f"{hotspot_stats.get('mean_lst_overall_C', 0):.2f}°C"),
            ("Heat Excess", f"+{hotspot_stats.get('lst_excess_C', 0):.2f}°C"),
            ("Max Gi* Z-score", f"{hotspot_stats.get('max_gi_star', 0):.3f}"),
        ]
        for j, (k, v) in enumerate(stats_items):
            y_pos = 0.88 - j * 0.09
            ax_tbl.text(0.05, y_pos, k + ":", color="#90CAF9", fontsize=10,
                        transform=ax_tbl.transAxes, fontweight="bold")
            ax_tbl.text(0.65, y_pos, v, color=WHITE, fontsize=10,
                        transform=ax_tbl.transAxes)
            if j < len(stats_items) - 1:
                ax_tbl.axhline(y=y_pos - 0.03, xmin=0.02, xmax=0.98,
                               color="#333", linewidth=0.5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 4: ML Model Validation ───────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        _page_header(fig, "Physics-Informed ML Model — Validation",
                     f"XGBoost + Energy Balance Constraint | R²={metrics.get('r2', 0):.4f}",
                     4, total_pages)
        fig.patch.set_facecolor(BG_DARK)

        ax_val = fig.add_axes([0.03, 0.10, 0.55, 0.80])
        _embed_image(ax_val, os.path.join(figs_dir, "model_validation.png"), "Validation: Actual vs Predicted LST")

        ax_met = fig.add_axes([0.60, 0.35, 0.38, 0.55])
        ax_met.set_facecolor(BG_CARD); ax_met.axis("off")
        ax_met.text(0.5, 0.96, "Model Metrics", ha="center", color=WHITE,
                    fontsize=12, fontweight="bold", transform=ax_met.transAxes)
        met_items = [
            ("R² Score",        f"{metrics.get('r2', 0):.4f}"),
            ("RMSE",            f"{metrics.get('rmse_C', 0):.3f}°C"),
            ("MAE",             f"{metrics.get('mae_C', 0):.3f}°C"),
            ("Physics Penalty", f"{metrics.get('physics_penalty', 0):.4f}"),
            ("PIML Score",      f"{metrics.get('piml_score', 0):.4f}"),
            ("Features",        "16"),
            ("Train Samples",   "32,000"),
            ("LSTM R²",         f"{lstm_metrics.get('r2', 0):.4f}"),
            ("LSTM RMSE",       f"{lstm_metrics.get('rmse_C', 0):.3f}°C"),
        ]
        for j, (k, v) in enumerate(met_items):
            y_pos = 0.88 - j * 0.10
            ax_met.text(0.05, y_pos, k + ":", color="#90CAF9", fontsize=10,
                        fontweight="bold", transform=ax_met.transAxes)
            ax_met.text(0.65, y_pos, v, color=WHITE, fontsize=10,
                        transform=ax_met.transAxes)

        ax_phys = fig.add_axes([0.60, 0.10, 0.38, 0.22])
        ax_phys.set_facecolor(BG_CARD); ax_phys.axis("off")
        ax_phys.text(0.5, 0.88, "Physics Constraint", ha="center", color=WHITE,
                     fontsize=11, fontweight="bold", transform=ax_phys.transAxes)
        phys_txt = "Rn = H + LE + G\nPenalty = λ × |Rn - H - LE - G|²\nλ = 0.15"
        ax_phys.text(0.5, 0.40, phys_txt, ha="center", va="center", color=LIGHT,
                     fontsize=10, transform=ax_phys.transAxes, family="monospace")

        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 5: Driver Analysis ────────────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        _page_header(fig, "Urban Heat Driver Analysis — SHAP",
                     "SHapley Additive exPlanations | Feature Importance Decomposition",
                     5, total_pages)
        fig.patch.set_facecolor(BG_DARK)

        ax_shap = fig.add_axes([0.03, 0.10, 0.55, 0.80])
        _embed_image(ax_shap, os.path.join(figs_dir, "shap_summary.png"), "SHAP Beeswarm Plot")

        ax_pie = fig.add_axes([0.60, 0.45, 0.38, 0.45])
        _embed_image(ax_pie, os.path.join(figs_dir, "driver_importance_bar.png"), "Top Drivers (% Contribution)")

        ax_tbl2 = fig.add_axes([0.60, 0.10, 0.38, 0.32])
        ax_tbl2.set_facecolor(BG_CARD); ax_tbl2.axis("off")
        ax_tbl2.text(0.5, 0.96, "Driver Rankings", ha="center", color=WHITE,
                     fontsize=11, fontweight="bold", transform=ax_tbl2.transAxes)
        if not driver_df.empty:
            for j, row in driver_df.head(5).iterrows():
                y_pos = 0.80 - j * 0.16
                ax_tbl2.text(0.05, y_pos, f"{int(row['rank'])}. {row['label'][:30]}",
                             color=WHITE, fontsize=9, transform=ax_tbl2.transAxes)
                ax_tbl2.text(0.80, y_pos, f"{row['pct_contribution']:.1f}%",
                             color="#FFB300", fontsize=9, fontweight="bold",
                             transform=ax_tbl2.transAxes)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 6: Cooling Scenarios ──────────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        _page_header(fig, "Cooling Scenario Simulation & Comparison",
                     "6 Intervention Types | 51.8 km² Coverage Each",
                     6, total_pages)
        fig.patch.set_facecolor(BG_DARK)

        ax_sc = fig.add_axes([0.03, 0.38, 0.94, 0.52])
        _embed_image(ax_sc, os.path.join(figs_dir, "scenario_comparison.png"),
                     "Cooling Effectiveness vs Cost Efficiency")

        ax_tbl3 = fig.add_axes([0.03, 0.08, 0.94, 0.27])
        ax_tbl3.set_facecolor(BG_CARD); ax_tbl3.axis("off")
        if not scenario_df.empty:
            cols_show = ["Intervention", "Area (km²)", "Mean ΔT (°C)", "Max ΔT (°C)", "Cost/°C"]
            cols_avail = [c for c in cols_show if c in scenario_df.columns]
            tbl_data = scenario_df[cols_avail].values
            col_labels = cols_avail
            tbl = ax_tbl3.table(
                cellText=tbl_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor(BLUE)
                    cell.set_text_props(color=WHITE, fontweight="bold")
                else:
                    cell.set_facecolor("#1a1f2e" if r % 2 == 0 else BG_CARD)
                    cell.set_text_props(color=WHITE)
                cell.set_edgecolor("#333")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 7: Optimal Strategy ───────────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        _page_header(fig, "Optimal Cooling Strategy",
                     f"Budget: {opt_results.get('budget_total',50000):,} units | "
                     f"Area: {opt_results.get('area_covered_km2',0)} km² | "
                     f"Cooling: -{opt_results.get('mean_cooling_hotspots_C',0):.2f}°C",
                     7, total_pages)
        fig.patch.set_facecolor(BG_DARK)

        ax_opt = fig.add_axes([0.03, 0.10, 0.65, 0.80])
        _embed_image(ax_opt, os.path.join(maps_dir, "optimal_strategy_map.png"),
                     "Baseline LST | Post-Intervention LST | Temperature Reduction (ΔT)")

        ax_rec = fig.add_axes([0.70, 0.10, 0.28, 0.80])
        ax_rec.set_facecolor(BG_CARD); ax_rec.axis("off")
        ax_rec.text(0.5, 0.97, "Recommendation", ha="center", color=WHITE,
                    fontsize=12, fontweight="bold", transform=ax_rec.transAxes)

        alloc = opt_results.get("intervention_allocation", {})
        rec_lines = [
            ("Budget Used",  f"{opt_results.get('budget_used',0):,.0f} units"),
            ("Budget Total", f"{opt_results.get('budget_total',50000):,} units"),
            ("Area Covered", f"{opt_results.get('area_covered_km2',0):.1f} km²"),
            ("Cells Treated",f"{opt_results.get('total_cells_allocated',0):,}"),
            ("Hot. Cooling", f"-{opt_results.get('mean_cooling_hotspots_C',0):.3f}°C"),
            ("City Cooling", f"-{opt_results.get('mean_cooling_city_C',0):.3f}°C"),
        ]
        for j, (k, v) in enumerate(rec_lines):
            y = 0.87 - j * 0.10
            ax_rec.text(0.05, y, k + ":", color="#90CAF9", fontsize=9,
                        fontweight="bold", transform=ax_rec.transAxes)
            ax_rec.text(0.60, y, v, color=WHITE, fontsize=9,
                        transform=ax_rec.transAxes)

        ax_rec.text(0.5, 0.27, "Intervention Mix:", ha="center", color=WHITE,
                    fontsize=10, fontweight="bold", transform=ax_rec.transAxes)
        for j2, (nm, cnt) in enumerate(alloc.items()):
            y2 = 0.19 - j2 * 0.09
            ax_rec.text(0.05, y2, f"• {nm.replace('_',' ').title()}",
                        color=GREEN, fontsize=9, transform=ax_rec.transAxes)
            ax_rec.text(0.85, y2, f"{cnt:,}", color=WHITE, fontsize=9,
                        transform=ax_rec.transAxes)

        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 8: LSTM Temporal Analysis ────────────────────────────────────
        fig = plt.figure(figsize=(11.7, 8.3))
        _page_header(fig, "Temporal Heat Dynamics — Physics-Informed LSTM",
                     f"365 days × 24h | R²={lstm_metrics.get('r2',0):.4f} | RMSE={lstm_metrics.get('rmse_C',0):.3f}°C",
                     8, total_pages)
        fig.patch.set_facecolor(BG_DARK)

        ax_lstm = fig.add_axes([0.03, 0.10, 0.94, 0.80])
        _embed_image(ax_lstm, os.path.join(figs_dir, "lstm_temporal_analysis.png"),
                     "Training Curves | 7-Day Prediction | Scatter | Diurnal LULC Profiles")

        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"  [SAVE] PDF report saved: {out_path}")
    return out_path
