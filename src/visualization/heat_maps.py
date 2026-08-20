"""
heat_maps.py
------------
Geospatial visualization of urban heat stress, hotspots, LST maps,
and intervention results using matplotlib and folium.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import os

from src.heat_analysis.heat_stress_index import (
    HEAT_STRESS_LABELS, HEAT_STRESS_COLORS
)


LULC_COLORS = {
    0: "#1565C0",   # Water - Dark Blue
    1: "#1B5E20",   # Dense Vegetation - Dark Green
    2: "#66BB6A",   # Sparse Vegetation - Light Green
    3: "#F9A825",   # Agriculture - Amber
    4: "#FFA726",   # Low-density Residential - Orange
    5: "#EF5350",   # High-density Residential - Red
    6: "#7B1FA2",   # Commercial/Industrial - Purple
    7: "#795548",   # Barren - Brown
}

LULC_LABELS = {
    0: "Water", 1: "Dense Vegetation", 2: "Sparse Vegetation",
    3: "Agriculture", 4: "Low-density Residential",
    5: "High-density Residential", 6: "Commercial/Industrial", 7: "Barren"
}


def plot_lulc_map(lulc: np.ndarray, config: dict, output_dir: str):
    """Plot Land Use / Land Cover classification map."""
    fig, ax = plt.subplots(figsize=(10, 9))

    lulc_classes = sorted(LULC_COLORS.keys())
    colors = [LULC_COLORS[c] for c in lulc_classes]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(lulc_classes + [max(lulc_classes) + 1], len(colors))

    im = ax.imshow(lulc, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"Land Use / Land Cover Map — {config['city']['name']}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Column (West → East)", fontsize=11)
    ax.set_ylabel("Row (North → South)", fontsize=11)

    patches = [mpatches.Patch(facecolor=LULC_COLORS[c],
                               label=f"{LULC_LABELS[c]}", edgecolor="gray")
               for c in lulc_classes]
    ax.legend(handles=patches, loc="lower left", fontsize=9,
              framealpha=0.9, title="LULC Classes", title_fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "lulc_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] LULC map saved: {path}")
    return path


def plot_lst_map(lst: np.ndarray, config: dict, output_dir: str,
                 title_suffix: str = "", filename: str = "lst_map.png"):
    """Plot Land Surface Temperature map."""
    fig, ax = plt.subplots(figsize=(10, 9))

    vmin, vmax = config["data"]["lst_range_celsius"]
    im = ax.imshow(lst, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, interpolation="bilinear")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Land Surface Temperature (°C)", fontsize=11)

    city = config["city"]["name"]
    ax.set_title(f"Land Surface Temperature — {city}{title_suffix}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Column (West → East)", fontsize=11)
    ax.set_ylabel("Row (North → South)", fontsize=11)

    # Add stats text box
    stats_text = (f"Mean: {lst.mean():.1f}°C\n"
                  f"Max:  {lst.max():.1f}°C\n"
                  f"Min:  {lst.min():.1f}°C\n"
                  f"Std:  {lst.std():.1f}°C")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] LST map saved: {path}")
    return path


def plot_heat_stress_map(stress_class: np.ndarray, config: dict, output_dir: str):
    """Plot UTCI-based heat stress classification map."""
    fig, ax = plt.subplots(figsize=(10, 9))

    stress_levels = list(range(6))
    colors = [HEAT_STRESS_COLORS[s] for s in stress_levels]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(stress_levels + [6], len(colors))

    ax.imshow(stress_class, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"Urban Heat Stress Map (UTCI) — {config['city']['name']}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Column (West → East)", fontsize=11)
    ax.set_ylabel("Row (North → South)", fontsize=11)

    patches = [mpatches.Patch(facecolor=HEAT_STRESS_COLORS[s],
                               label=HEAT_STRESS_LABELS[s], edgecolor="gray")
               for s in stress_levels]
    ax.legend(handles=patches, loc="lower left", fontsize=9,
              framealpha=0.9, title="Heat Stress Level", title_fontsize=10)

    # Add fraction stats
    total = stress_class.size
    stats_lines = []
    for s in [3, 4, 5]:
        frac = (stress_class == s).sum() / total * 100
        if frac > 0:
            stats_lines.append(f"{HEAT_STRESS_LABELS[s]}: {frac:.1f}%")
    if stats_lines:
        ax.text(0.98, 0.98, "\n".join(stats_lines), transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "heat_stress_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] Heat stress map saved: {path}")
    return path


def plot_hotspot_map(hotspot_class: np.ndarray, gi_star: np.ndarray,
                     config: dict, output_dir: str):
    """Plot Getis-Ord Gi* hotspot classification map."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: Classified hotspots
    colors = ["#1565C0", "#42A5F5", "#E0E0E0", "#EF9A9A", "#C62828"]
    labels = ["Sig. Cold Spot (p<0.01)", "Cold Spot (p<0.05)",
              "Not Significant",
              "Hot Spot (p<0.05)", "Sig. Hot Spot (p<0.01)"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], 5)

    axes[0].imshow(hotspot_class, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("Getis-Ord Gi* Hotspot Classification", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("West → East")
    axes[0].set_ylabel("North → South")

    patches = [mpatches.Patch(facecolor=colors[i], label=labels[i], edgecolor="gray")
               for i in range(5)]
    axes[0].legend(handles=patches, loc="lower left", fontsize=8, framealpha=0.9)

    # Panel 2: Gi* Z-score surface
    vmax = max(abs(gi_star.min()), abs(gi_star.max()))
    im2 = axes[1].imshow(gi_star, cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, interpolation="bilinear")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Getis-Ord Gi* Z-Score")
    axes[1].set_title("Gi* Z-Score Surface\n(Positive = Heat Cluster)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("West → East")
    axes[1].set_ylabel("North → South")

    plt.suptitle(f"Urban Heat Hotspot Detection — {config['city']['name']}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "hotspot_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] Hotspot map saved: {path}")
    return path


def plot_optimal_strategy_map(lst_baseline: np.ndarray,
                               lst_optimal: np.ndarray,
                               delta_lst: np.ndarray,
                               allocation_map: dict,
                               config: dict,
                               output_dir: str):
    """
    3-panel map showing: Baseline LST | Optimal LST | Temperature Reduction.
    """
    from src.cooling_scenarios.interventions import INTERVENTIONS
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    rows, cols = lst_baseline.shape
    vmin = config["data"]["lst_range_celsius"][0]
    vmax = config["data"]["lst_range_celsius"][1]

    # Panel 1: Baseline LST
    im1 = axes[0].imshow(lst_baseline, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                          interpolation="bilinear")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="LST (°C)")
    axes[0].set_title("Baseline LST", fontsize=13, fontweight="bold")

    # Panel 2: Optimal LST
    im2 = axes[1].imshow(lst_optimal, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                          interpolation="bilinear")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="LST (°C)")
    axes[1].set_title("LST After Optimal Interventions", fontsize=13, fontweight="bold")

    # Panel 3: ΔT (cooling)
    max_delta = max(abs(delta_lst.min()), abs(delta_lst.max())) + 0.5
    im3 = axes[2].imshow(-delta_lst, cmap="Blues", vmin=0, vmax=max_delta,
                          interpolation="bilinear")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04,
                 label="Temperature Reduction (°C)")
    axes[2].set_title("Temperature Reduction Map\n(Darker Blue = More Cooling)",
                      fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("West → East", fontsize=10)
        ax.set_ylabel("North → South", fontsize=10)

    plt.suptitle(f"Optimal Cooling Strategy — {config['city']['name']}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "optimal_strategy_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] Optimal strategy map saved: {path}")
    return path

