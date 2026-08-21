"""
app.py  —  Urban Heat Mitigation Interactive Dashboard
-------------------------------------------------------
Streamlit web app providing:
  - Live city heat stress map viewer
  - Interactive intervention controls
  - Real-time scenario simulation
  - Driver analysis (SHAP)
  - Optimal strategy explorer
  - Downloadable reports

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import time
import warnings

# Fix Windows encoding for any background libraries (like SHAP) that print progress bars
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
import io

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Heat Mitigation AI/ML",
    page_icon="🌡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    .hotspot-badge {
        background: #c62828; color: white; border-radius: 8px;
        padding: 4px 10px; font-weight: bold; font-size: 0.9rem;
    }
    .cooling-badge {
        background: #1565C0; color: white; border-radius: 8px;
        padding: 4px 10px; font-weight: bold; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Pipeline Functions ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Generating city data...")
def load_pipeline():
    """Load and cache the full pipeline (run once per session)."""
    from src.data_pipeline.synthetic_data import load_config, grid_to_dataframe
    from src.data_pipeline.process_real_data import load_real_city_grid
    from src.heat_analysis.heat_stress_index import utci_approximation, wbgt_outdoor, classify_heat_stress
    from src.heat_analysis.hotspot_detector import detect_hotspots
    from src.ml_models.piml_model import train_piml_pipeline, predict_full_grid

    config = load_config("config.yaml")
    data = load_real_city_grid(config)
    
    # FIX: Ensure dictionary keys exactly match what grid_to_dataframe expects
    if "lat" in data: data["lat_grid"] = data.pop("lat")
    if "lon" in data: data["lon_grid"] = data.pop("lon")
    if "is_water" in data: data["water_mask"] = data.pop("is_water")
    
    df = grid_to_dataframe(data)

    utci = utci_approximation(data["air_temp"], data["humidity"], data["lst"], data["wind_speed"])
    wbgt = wbgt_outdoor(data["air_temp"], data["humidity"], data["lst"], data["wind_speed"])
    stress_class = classify_heat_stress(utci)

    hotspot_results = detect_hotspots(data["lst"], utci, config)
    ml_results = train_piml_pipeline(df, config)

    model = ml_results["model"]
    feature_cols = ml_results["feature_cols"]
    lst_predicted_flat = predict_full_grid(model, df, feature_cols)
    lst_predicted = lst_predicted_flat.reshape(data["rows"], data["cols"])

    return {
        "config": config,
        "data": data,
        "df": df,
        "utci": utci,
        "wbgt": wbgt,
        "stress_class": stress_class,
        "hotspot_results": hotspot_results,
        "ml_results": ml_results,
        "model": model,
        "feature_cols": feature_cols,
        "lst_predicted": lst_predicted,
    }


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def make_lst_map(lst, config, title="LST (°C)", cmap="RdYlGn_r"):
    fig, ax = plt.subplots(figsize=(7, 6))
    vmin, vmax = config["data"]["lst_range_celsius"]
    im = ax.imshow(lst, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="°C")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("West → East"); ax.set_ylabel("North → South")
    stats = f"Mean: {lst.mean():.1f}°C  Max: {lst.max():.1f}°C"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    return fig


def make_hotspot_map(hotspot_class, gi_star):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#1565C0", "#42A5F5", "#E0E0E0", "#EF9A9A", "#C62828"]
    labels = ["Sig. Cold (p<0.01)", "Cold (p<0.05)", "Not Sig.",
              "Hot (p<0.05)", "Sig. Hot (p<0.01)"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], 5)
    axes[0].imshow(hotspot_class, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("Getis-Ord Gi* Classification", fontsize=11, fontweight="bold")
    patches = [mpatches.Patch(facecolor=colors[i], label=labels[i], edgecolor="gray")
               for i in range(5)]
    axes[0].legend(handles=patches, loc="lower left", fontsize=7, framealpha=0.9)
    vmax = max(abs(gi_star.min()), abs(gi_star.max()))
    im2 = axes[1].imshow(gi_star, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Gi* Z-Score")
    axes[1].set_title("Gi* Z-Score Surface", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def make_stress_map(stress_class):
    from src.heat_analysis.heat_stress_index import HEAT_STRESS_LABELS, HEAT_STRESS_COLORS
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [HEAT_STRESS_COLORS[s] for s in range(6)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(range(7)), 6)
    ax.imshow(stress_class, cmap=cmap, norm=norm, interpolation="nearest")
    patches = [mpatches.Patch(facecolor=HEAT_STRESS_COLORS[s], label=HEAT_STRESS_LABELS[s])
               for s in range(6)]
    ax.legend(handles=patches, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("UTCI Heat Stress Classification", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def run_live_scenario(intervention_name, ndvi_delta, albedo_delta, humidity_delta,
                      coverage_pct, pipeline):
    """Run a custom intervention scenario with user-defined deltas."""
    from src.cooling_scenarios.interventions import get_hotspot_application_mask, Intervention

    model = pipeline["model"]
    feature_cols = pipeline["feature_cols"]
    df = pipeline["df"]
    lst_predicted = pipeline["lst_predicted"]
    hotspot_class = pipeline["hotspot_results"]["hotspot_class"]
    lulc_grid = pipeline["data"]["lulc"]
    config = pipeline["config"]

    rows, cols = lst_predicted.shape

    custom_intv = Intervention(
        name="custom",
        label=f"Custom: {intervention_name}",
        ndvi_delta=ndvi_delta,
        albedo_delta=albedo_delta,
        humidity_delta=humidity_delta,
        cost_per_cell=10,
        applicable_lulc=[4, 5, 6, 7],
    )

    mask = get_hotspot_application_mask(
        hotspot_class, lulc_grid, custom_intv,
        coverage_fraction=coverage_pct / 100.0
    )

    df_mod = custom_intv.apply(df, mask)
    X_mod = df_mod[feature_cols].astype(float)
    lst_new_flat = model.predict(X_mod)
    lst_new = lst_new_flat.reshape(rows, cols)

    delta = lst_new - lst_predicted
    hot_mask = (hotspot_class >= 1)
    mean_cooling = float(-delta[hot_mask].mean()) if hot_mask.any() else 0.0
    max_cooling = float(-delta.min())
    cells = int(mask.sum())
    area = cells * (config["grid"]["resolution_m"] / 1000.0) ** 2

    return {
        "lst_new": lst_new,
        "delta": delta,
        "mean_cooling": round(mean_cooling, 3),
        "max_cooling": round(max_cooling, 3),
        "cells": cells,
        "area_km2": round(area, 1),
    }


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a237e, #0d47a1, #01579b);
                padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;">
        <h1 style="margin:0; font-size:2.2rem;">🌡 Urban Heat Mitigation AI/ML System</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.85; font-size:1.1rem;">
            Geospatial Physics-Informed Analysis & Cooling Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load pipeline (cached) ────────────────────────────────────────────────
    with st.spinner("Initializing pipeline (first load ~30s, cached thereafter)..."):
        pipeline = load_pipeline()

    config = pipeline["config"]
    data = pipeline["data"]
    utci = pipeline["utci"]
    stress_class = pipeline["stress_class"]
    hotspot_results = pipeline["hotspot_results"]
    ml_results = pipeline["ml_results"]
    lst_predicted = pipeline["lst_predicted"]
    hs = hotspot_results["hotspot_stats"]
    metrics = ml_results["metrics"]

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Mean UTCI", f"{utci.mean():.1f}°C", "Heat Stress Index"),
        ("Extreme Stress", f"{(stress_class >= 4).mean()*100:.1f}%", "of city area"),
        ("Hotspot Cells", f"{hs['n_hotspot_cells']:,}", f"{hs['hotspot_fraction_pct']}% of city"),
        ("Heat Excess", f"+{hs['lst_excess_C']}°C", "hotspot vs city mean"),
        ("Model R²", f"{metrics['r2']:.4f}", f"RMSE={metrics['rmse_C']:.3f}°C"),
    ]
    for col, (label, value, sub) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🗺 Maps", "🔥 Hotspots", "📊 Drivers", "❄ Scenarios", "🧬 Optimizer", "📋 Report"
    ])

    # ── Tab 1: Maps ───────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Urban Heat Maps")
        map_col1, map_col2 = st.columns(2)

        with map_col1:
            st.markdown("#### Land Surface Temperature")
            fig = make_lst_map(data["lst"], config, "Baseline LST (Real Landsat-9)")
            st.image(fig_to_bytes(fig), use_container_width=True)

            st.markdown("#### UTCI Heat Stress")
            fig2 = make_stress_map(stress_class)
            st.image(fig_to_bytes(fig2), use_container_width=True)

        with map_col2:
            st.markdown("#### ML-Predicted LST")
            fig3 = make_lst_map(lst_predicted, config, f"Physics-Informed XGBoost (R²={metrics['r2']:.4f})")
            st.image(fig_to_bytes(fig3), use_container_width=True)

            st.markdown("#### NDVI Distribution")
            fig4, ax4 = plt.subplots(figsize=(7, 6))
            im4 = ax4.imshow(data["ndvi"], cmap="YlGn", vmin=-0.1, vmax=0.85, interpolation="bilinear")
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="NDVI")
            ax4.set_title("Vegetation Index (NDVI)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.image(fig_to_bytes(fig4), use_container_width=True)

        # LST histogram
        st.markdown("#### LST Distribution")
        fig5, ax5 = plt.subplots(figsize=(12, 3))
        ax5.hist(data["lst"].ravel(), bins=80, color="#E53935", alpha=0.7, label="Baseline LST", edgecolor="white")
        ax5.hist(lst_predicted.ravel(), bins=80, color="#1565C0", alpha=0.6, label="ML-Predicted LST", edgecolor="white")
        ax5.set_xlabel("Temperature (°C)"); ax5.set_ylabel("Count")
        ax5.set_title("LST Distribution — Baseline vs ML Predicted")
        ax5.legend(); ax5.grid(alpha=0.3)
        plt.tight_layout()
        st.image(fig_to_bytes(fig5), use_container_width=True)

    # ── Tab 2: Hotspots ───────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Urban Heat Hotspot Detection — Getis-Ord Gi*")

        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Hotspot Cells", f"{hs['n_hotspot_cells']:,}", f"{hs['hotspot_fraction_pct']}%")
        hc2.metric("Mean LST (Hotspots)", f"{hs['mean_lst_hotspot_C']}°C",
                   f"+{hs['lst_excess_C']}°C excess")
        hc3.metric("Max Gi* Z-Score", f"{hs['max_gi_star']:.2f}", "High = strong cluster")

        fig_hs = make_hotspot_map(
            hotspot_results["hotspot_class"],
            hotspot_results["gi_star_lst"]
        )
        st.image(fig_to_bytes(fig_hs), use_container_width=True)

        # KDE surface
        st.markdown("#### KDE Hotspot Intensity Surface")
        fig_kde, ax_kde = plt.subplots(figsize=(10, 5))
        im_kde = ax_kde.imshow(hotspot_results["kde_surface"], cmap="inferno", interpolation="bilinear")
        plt.colorbar(im_kde, ax=ax_kde, label="Hotspot Intensity")
        ax_kde.set_title("Kernel Density Estimation — Heat Hotspot Probability", fontweight="bold")
        plt.tight_layout()
        st.image(fig_to_bytes(fig_kde), use_container_width=True)

    # ── Tab 3: Driver Analysis ────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Urban Heat Driver Analysis (SHAP)")

        with st.spinner("Computing SHAP values..."):
            @st.cache_data
            def get_driver_results(_model, _X_train, feature_cols):
                from src.heat_analysis.driver_analysis import analyze_drivers
                return analyze_drivers(_model, _X_train, feature_cols, n_shap_samples=2000)

            driver_results = get_driver_results(
                pipeline["model"],
                ml_results["X_train"],
                pipeline["feature_cols"],
            )

        dc1, dc2 = st.columns([3, 2])

        with dc1:
            st.markdown("#### Feature Importance (% SHAP Contribution)")
            top_df = driver_results["importance_df"].head(10)
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            top_rev = top_df.iloc[::-1]
            ax_bar.barh(top_rev["label"], top_rev["pct_contribution"],
                        color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top_rev))),
                        edgecolor="white")
            for i, val in enumerate(top_rev["pct_contribution"]):
                ax_bar.text(val + 0.2, i, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
            ax_bar.set_xlabel("% Contribution (SHAP)")
            ax_bar.set_title("Top 10 Urban Heat Drivers")
            ax_bar.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.image(fig_to_bytes(fig_bar), use_container_width=True)

        with dc2:
            st.markdown("#### Category Breakdown")
            grouped = driver_results["grouped_df"]
            cat_colors = {
                "Vegetation": "#2E7D32", "Surface Properties": "#F57F17",
                "Urban Morphology": "#7B1FA2", "Atmospheric": "#0288D1",
                "Land Use / Land Cover": "#D84315", "Geographic": "#546E7A",
            }
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            colors = [cat_colors.get(c, "#9E9E9E") for c in grouped["category"]]
            ax_pie.pie(grouped["pct_contribution"], labels=grouped["category"],
                       colors=colors, autopct="%1.1f%%", startangle=140,
                       wedgeprops=dict(edgecolor="white", linewidth=2))
            ax_pie.set_title("Driver Categories")
            plt.tight_layout()
            st.image(fig_to_bytes(fig_pie), use_container_width=True)

            st.markdown("**Category Contributions:**")
            for _, row in grouped.iterrows():
                st.progress(int(row["pct_contribution"]),
                            text=f"{row['category']}: {row['pct_contribution']:.1f}%")

    # ── Tab 4: Cooling Scenarios ──────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Cooling Scenario Simulation")

        st.markdown("#### Standard Scenario Comparison")

        @st.cache_data
        def get_scenario_results(_model, _df, _lst_predicted, feature_cols, _hotspot_class, _lulc, _config):
            from src.cooling_scenarios.scenario_engine import run_all_scenarios
            return run_all_scenarios(
                df_baseline=_df, lst_baseline=_lst_predicted,
                model=_model, feature_cols=feature_cols,
                hotspot_class=_hotspot_class, lulc_grid=_lulc, config=_config,
            )

        scenario_results = get_scenario_results(
            pipeline["model"], pipeline["df"], lst_predicted,
            pipeline["feature_cols"], hotspot_results["hotspot_class"],
            data["lulc"], config,
        )
        st.dataframe(scenario_results["summary_df"], use_container_width=True)

        # Scenario bar chart
        fig_sc, (ax_sc1, ax_sc2) = plt.subplots(1, 2, figsize=(14, 5))
        names = [r["label"] for r in scenario_results["scenarios"].values()]
        coolings = [r["mean_cooling_C"] for r in scenario_results["scenarios"].values()]
        costs = [r["cost_per_degree_C"] for r in scenario_results["scenarios"].values()]
        clrs = [r["color"] for r in scenario_results["scenarios"].values()]

        ax_sc1.barh(names, coolings, color=clrs, edgecolor="white")
        ax_sc1.set_xlabel("Mean Cooling (°C)"); ax_sc1.set_title("Cooling Effectiveness")
        for bar, v in zip(ax_sc1.patches, coolings):
            ax_sc1.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                        f"-{v:.2f}°C", va="center", fontsize=9, fontweight="bold")
        ax_sc1.grid(axis="x", alpha=0.3)

        ax_sc2.barh(names, costs, color=clrs, edgecolor="white")
        ax_sc2.set_xlabel("Cost per °C (budget units)"); ax_sc2.set_title("Cost Efficiency (Lower = Better)")
        ax_sc2.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.image(fig_to_bytes(fig_sc), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎛 Custom Intervention Simulator")
        st.info("Adjust the sliders to design a custom cooling intervention and see predicted temperature reduction in real-time.")

        sim_c1, sim_c2 = st.columns([1, 2])
        with sim_c1:
            intv_name = st.text_input("Intervention Name", value="My Intervention")
            ndvi_delta = st.slider("NDVI Increase", 0.0, 0.5, 0.20, 0.05,
                                   help="Higher NDVI = more vegetation cooling")
            albedo_delta = st.slider("Albedo Change", -0.15, 0.35, 0.10, 0.05,
                                     help="Higher albedo = more solar reflection")
            humidity_delta = st.slider("Humidity Forcing", 0.0, 0.5, 0.10, 0.05,
                                       help="Evaporative cooling contribution")
            coverage_pct = st.slider("Coverage of Hotspot Cells (%)", 10, 100, 60, 10)

            if st.button("Run Simulation", type="primary"):
                with st.spinner("Simulating..."):
                    sim_result = run_live_scenario(
                        intv_name, ndvi_delta, albedo_delta, humidity_delta,
                        coverage_pct, pipeline
                    )
                    st.session_state["sim_result"] = sim_result

        with sim_c2:
            if "sim_result" in st.session_state:
                res = st.session_state["sim_result"]
                r1, r2, r3 = st.columns(3)
                r1.metric("Mean Cooling", f"-{res['mean_cooling']}°C")
                r2.metric("Max Cooling", f"-{res['max_cooling']}°C")
                r3.metric("Area Covered", f"{res['area_km2']} km²")

                fig_sim, axes = plt.subplots(1, 2, figsize=(12, 5))
                vmin, vmax = config["data"]["lst_range_celsius"]
                im0 = axes[0].imshow(lst_predicted, cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
                plt.colorbar(im0, ax=axes[0], label="LST (°C)")
                axes[0].set_title("Baseline LST", fontweight="bold")

                max_delta = max(abs(res["delta"].min()), abs(res["delta"].max())) + 0.1
                im1 = axes[1].imshow(-res["delta"], cmap="Blues", vmin=0, vmax=max_delta)
                plt.colorbar(im1, ax=axes[1], label="Cooling (°C)")
                axes[1].set_title(f"Temperature Reduction — {intv_name}", fontweight="bold")
                plt.tight_layout()
                st.image(fig_to_bytes(fig_sim), use_container_width=True)
            else:
                st.info("Set parameters and click 'Run Simulation' to see results.")

    # ── Tab 5: Optimizer ──────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Budget-Constrained Cooling Optimizer")

        opt_c1, opt_c2 = st.columns([1, 2])

        with opt_c1:
            budget = st.number_input("Budget (units)", min_value=5000, max_value=500000,
                                      value=50000, step=5000)
            max_coverage = st.slider("Max Coverage Fraction", 0.1, 0.6, 0.35, 0.05)
            run_opt = st.button("Run Optimization", type="primary")

        if run_opt or "opt_results" in st.session_state:
            if run_opt:
                from src.cooling_scenarios.optimizer import greedy_optimize
                mod_config = dict(config)
                mod_config["optimization"] = dict(config["optimization"])
                mod_config["optimization"]["budget_units"] = budget
                mod_config["optimization"]["max_coverage_fraction"] = max_coverage

                with st.spinner("Running optimization..."):
                    opt_results = greedy_optimize(
                        df_baseline=pipeline["df"],
                        lst_baseline=lst_predicted,
                        model=pipeline["model"],
                        feature_cols=pipeline["feature_cols"],
                        hotspot_class=hotspot_results["hotspot_class"],
                        lulc_grid=data["lulc"],
                        config=mod_config,
                    )
                st.session_state["opt_results"] = opt_results

            opt = st.session_state["opt_results"]

            with opt_c2:
                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.metric("Budget Used", f"{opt['budget_used']:,.0f}")
                oc2.metric("Area Covered", f"{opt['area_covered_km2']} km²")
                oc3.metric("Hotspot Cooling", f"-{opt['mean_cooling_hotspots_C']}°C")
                oc4.metric("City Cooling", f"-{opt['mean_cooling_city_C']}°C")

            st.markdown("**Intervention Allocation:**")
            for intv_name, count in opt["intervention_allocation"].items():
                st.markdown(f"- **{intv_name}**: {count:,} cells")

            # Strategy map
            fig_opt, axes = plt.subplots(1, 3, figsize=(18, 5))
            vmin_lst, vmax_lst = config["data"]["lst_range_celsius"]

            im0 = axes[0].imshow(lst_predicted, cmap="RdYlGn_r", vmin=vmin_lst, vmax=vmax_lst)
            plt.colorbar(im0, ax=axes[0], label="LST (°C)")
            axes[0].set_title("Baseline LST", fontweight="bold")

            im1 = axes[1].imshow(opt["lst_optimal"], cmap="RdYlGn_r", vmin=vmin_lst, vmax=vmax_lst)
            plt.colorbar(im1, ax=axes[1], label="LST (°C)")
            axes[1].set_title("Post-Intervention LST", fontweight="bold")

            max_d = max(abs(opt["delta_optimal"].min()), 0.01)
            im2 = axes[2].imshow(-opt["delta_optimal"], cmap="Blues", vmin=0, vmax=max_d)
            plt.colorbar(im2, ax=axes[2], label="Cooling (°C)")
            axes[2].set_title("Temperature Reduction Map", fontweight="bold")

            plt.suptitle("Optimal Cooling Strategy", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.image(fig_to_bytes(fig_opt), use_container_width=True)
        else:
            st.info("Set budget and click 'Run Optimization'.")

    # ── Tab 6: Report ─────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Final Analysis Report")

        report_path = "outputs/reports/final_report.json"
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)

            st.json(report)

            with open(report_path) as f:
                report_bytes = f.read()
            st.download_button(
                label="Download JSON Report",
                data=report_bytes,
                file_name="urban_heat_report.json",
                mime="application/json",
            )

        # Download scenario CSV
        scen_path = "outputs/reports/scenario_summary.csv"
        if os.path.exists(scen_path):
            with open(scen_path) as f:
                csv_bytes = f.read()
            st.download_button(
                label="Download Scenario Summary CSV",
                data=csv_bytes,
                file_name="scenario_summary.csv",
                mime="text/csv",
            )

        driver_path = "outputs/reports/driver_importance.csv"
        if os.path.exists(driver_path):
            df_drv = pd.read_csv(driver_path)
            st.markdown("#### Driver Importance Table")
            st.dataframe(df_drv[["rank", "label", "pct_contribution"]].head(15),
                         use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:0.85rem;'>"
        "Urban Heat Mitigation AI/ML System &nbsp;|&nbsp; "
        "Physics-Informed XGBoost + SHAP + Getis-Ord Gi* + Greedy Optimizer"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()