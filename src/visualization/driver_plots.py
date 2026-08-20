"""
driver_plots.py
---------------
SHAP-based visualization for urban heat driver analysis.
Produces summary plots, waterfall charts, and grouped category plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import shap
import os


def plot_shap_summary(shap_values: np.ndarray,
                      X_sample: pd.DataFrame,
                      feature_names: list,
                      output_dir: str,
                      max_display: int = 15):
    """Plot SHAP beeswarm summary (feature importance + direction)."""
    # Map feature names to readable labels
    from src.heat_analysis.driver_analysis import FEATURE_LABELS
    readable_names = [FEATURE_LABELS.get(f, f) for f in feature_names]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Compute mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:max_display]

    top_shap = shap_values[:, top_idx]
    top_feat = X_sample.iloc[:, top_idx]
    top_names = [readable_names[i] for i in top_idx]
    top_means = mean_abs_shap[top_idx]

    # Color by feature value (normalized)
    colors = plt.cm.RdYlGn_r

    for rank, (fi, fname, mean_val) in enumerate(zip(top_idx, top_names, top_means)):
        y_vals = np.random.normal(rank, 0.08, size=len(shap_values))
        feat_vals = X_sample.iloc[:, fi].values
        feat_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-10)

        scatter = ax.scatter(shap_values[:, fi], y_vals,
                             c=feat_norm, cmap=colors,
                             alpha=0.4, s=8, vmin=0, vmax=1)

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (Impact on LST °C)", fontsize=12)
    ax.set_title("SHAP Feature Importance — Urban Heat Drivers\n"
                 "(Color: Feature value Low→High)", fontsize=13, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=colors)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Feature Value", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] SHAP summary saved: {path}")
    return path


def plot_driver_importance_bar(importance_df: pd.DataFrame, output_dir: str):
    """Horizontal bar chart of feature importance (% contribution)."""
    top = importance_df.head(12).copy()
    top = top.iloc[::-1]  # Reverse for bottom-to-top display

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(range(len(top)), top["pct_contribution"],
                   color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top))),
                   edgecolor="white", height=0.7)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=10)
    ax.set_xlabel("% Contribution to Urban Heat (SHAP)", fontsize=12)
    ax.set_title("Key Drivers of Urban Land Surface Temperature\n(SHAP-based analysis)",
                 fontsize=13, fontweight="bold")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top["pct_contribution"])):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, top["pct_contribution"].max() * 1.2)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "driver_importance_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] Driver importance bar chart saved: {path}")
    return path


def plot_driver_category_pie(grouped_df: pd.DataFrame, output_dir: str):
    """Pie chart of driver contribution by category."""
    category_colors = {
        "Vegetation":             "#2E7D32",
        "Surface Properties":     "#F57F17",
        "Urban Morphology":       "#7B1FA2",
        "Atmospheric":            "#0288D1",
        "Land Use / Land Cover":  "#D84315",
        "Geographic":             "#546E7A",
        "Other":                  "#9E9E9E",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Pie chart
    colors = [category_colors.get(cat, "#9E9E9E") for cat in grouped_df["category"]]
    wedges, texts, autotexts = ax1.pie(
        grouped_df["pct_contribution"],
        labels=grouped_df["category"],
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.8,
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
    ax1.set_title("Driver Category Contributions\nto Urban Heat (SHAP)",
                  fontsize=13, fontweight="bold")

    # Horizontal bar of categories
    cat_rev = grouped_df.iloc[::-1]
    colors_rev = [category_colors.get(cat, "#9E9E9E") for cat in cat_rev["category"]]
    ax2.barh(cat_rev["category"], cat_rev["pct_contribution"],
             color=colors_rev, edgecolor="white", height=0.6)
    for i, val in enumerate(cat_rev["pct_contribution"]):
        ax2.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("% Contribution (SHAP)", fontsize=12)
    ax2.set_title("Driver Categories Ranked", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle("Urban Heat Driver Analysis — SHAP Decomposition",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "driver_category_pie.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] Driver category chart saved: {path}")
    return path


def plot_scenario_comparison(scenarios: dict, output_dir: str):
    """Bar chart comparing cooling effectiveness of all scenarios."""
    names = []
    coolings = []
    costs = []
    colors = []

    for name, r in scenarios.items():
        names.append(r["label"])
        coolings.append(r["mean_cooling_C"])
        costs.append(r["cost_per_degree_C"])
        colors.append(r["color"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Cooling effectiveness
    bars = ax1.barh(names, coolings, color=colors, edgecolor="white", height=0.6)
    ax1.set_xlabel("Mean Temperature Reduction (°C)", fontsize=12)
    ax1.set_title("Cooling Effectiveness\nby Intervention", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, coolings):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"-{val:.2f}°C", va="center", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, max(coolings) * 1.25)
    ax1.grid(axis="x", alpha=0.3)
    ax1.axvline(0, color="black", linewidth=0.5)

    # Cost efficiency
    bars2 = ax2.barh(names, costs, color=colors, edgecolor="white", height=0.6)
    ax2.set_xlabel("Cost per °C Reduction (budget units/°C)", fontsize=12)
    ax2.set_title("Cost Efficiency\n(Lower = Better)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, costs):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}", va="center", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle("Cooling Scenario Comparison — All Interventions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "scenario_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] Scenario comparison saved: {path}")
    return path


def plot_model_validation(y_test: np.ndarray, y_pred: np.ndarray,
                           metrics: dict, output_dir: str):
    """Scatter plot of predicted vs actual LST with regression line."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.15, s=3, color="#1565C0")
    lim = [min(y_test.min(), y_pred.min()) - 1,
           max(y_test.max(), y_pred.max()) + 1]
    ax.plot(lim, lim, "r--", linewidth=2, label="1:1 line")
    ax.set_xlabel("Actual LST (°C)", fontsize=12)
    ax.set_ylabel("Predicted LST (°C)", fontsize=12)
    ax.set_title("Model Validation: Actual vs Predicted LST", fontsize=13, fontweight="bold")
    ax.set_xlim(lim); ax.set_ylim(lim)

    stats_text = (f"R² = {metrics['r2']:.4f}\n"
                  f"RMSE = {metrics['rmse_C']:.3f}°C\n"
                  f"MAE = {metrics['mae_C']:.3f}°C\n"
                  f"PIML Score = {metrics['piml_score']:.4f}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Residual distribution
    ax2 = axes[1]
    residuals = y_pred - y_test
    ax2.hist(residuals, bins=60, color="#1565C0", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Residual (Predicted − Actual) °C", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Residual Distribution", fontsize=13, fontweight="bold")
    ax2.text(0.05, 0.95, f"Mean: {residuals.mean():.3f}°C\nStd: {residuals.std():.3f}°C",
             transform=ax2.transAxes, verticalalignment="top", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "model_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] Model validation plot saved: {path}")
    return path

