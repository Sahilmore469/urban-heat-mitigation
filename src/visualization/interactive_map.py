"""
interactive_map.py
------------------
Generates interactive Folium HTML maps with:
  - Choropleth LST layer
  - Hotspot circle markers with popups
  - Intervention placement markers
  - Layer control for toggling
  - Color-coded heat stress zones

The map is centered on the city coordinates from config.yaml.
"""

import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import os


def make_interactive_heat_map(
    lst: np.ndarray,
    hotspot_class: np.ndarray,
    stress_class: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    opt_allocation: dict,
    lulc: np.ndarray,
    config: dict,
    output_dir: str,
) -> str:
    """
    Build a multi-layer Folium interactive map.

    Layers:
      1. LST HeatMap   – continuous surface of land surface temperature
      2. Hotspot Markers – Gi* significant hotspot centroids
      3. Optimal Intervention – where urban greening etc. was placed
      4. UTCI Stress Zones – color overlays by heat stress level

    Returns path to saved HTML file.
    """
    city = config["city"]
    center = [city["center_lat"], city["center_lon"]]
    rows, cols = lst.shape

    m = folium.Map(
        location=center,
        zoom_start=12,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # ── Add tile options ─────────────────────────────────────────────────────
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Matter").add_to(m)

    # ── 1. LST HeatMap Layer ─────────────────────────────────────────────────
    # Sample every 4th cell for performance
    step = 4
    heat_data = []
    lst_min, lst_max = lst.min(), lst.max()

    for r in range(0, rows, step):
        for c in range(0, cols, step):
            lat = float(lat_grid[r, c])
            lon = float(lon_grid[r, c])
            intensity = float((lst[r, c] - lst_min) / (lst_max - lst_min + 1e-6))
            heat_data.append([lat, lon, intensity])

    HeatMap(
        heat_data,
        name="Land Surface Temperature",
        min_opacity=0.3,
        max_zoom=18,
        radius=18,
        blur=15,
        gradient={
            "0.0": "#2196F3",   # Cool blue
            "0.3": "#4CAF50",   # Green
            "0.5": "#FFEB3B",   # Yellow
            "0.7": "#FF9800",   # Orange
            "0.9": "#F44336",   # Red
            "1.0": "#7B1FA2",   # Purple (extreme)
        },
    ).add_to(m)

    # ── 2. Hotspot Markers ───────────────────────────────────────────────────
    hotspot_group = folium.FeatureGroup(name="Heat Hotspots (Gi*)", show=True)

    hot_class_flat = hotspot_class.ravel()
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    lst_flat = lst.ravel()
    lulc_flat = lulc.ravel()

    lulc_labels = {
        0: "Water", 1: "Dense Vegetation", 2: "Sparse Vegetation",
        3: "Agriculture", 4: "Low-density Residential",
        5: "High-density Residential", 6: "Commercial/Industrial", 7: "Barren",
    }

    # Only show significant hotspot cells, sampled for performance
    hot_indices = np.where(hot_class_flat >= 2)[0]
    sample_step = max(1, len(hot_indices) // 200)  # Show max ~200 markers

    for idx in hot_indices[::sample_step]:
        lat_pt = float(lat_flat[idx])
        lon_pt = float(lon_flat[idx])
        lst_val = float(lst_flat[idx])
        lulc_val = int(lulc_flat[idx])

        popup_html = f"""
        <div style="font-family:Arial; min-width:160px;">
          <b style="color:#c62828;">SIGNIFICANT HOT SPOT</b><br>
          <hr style="margin:4px 0">
          <b>LST:</b> {lst_val:.1f}°C<br>
          <b>LULC:</b> {lulc_labels.get(lulc_val, 'Unknown')}<br>
          <b>Gi* Class:</b> Significant (p&lt;0.01)<br>
          <b>Coords:</b> {lat_pt:.4f}°N, {lon_pt:.4f}°E
        </div>"""

        folium.CircleMarker(
            location=[lat_pt, lon_pt],
            radius=5,
            color="#c62828",
            fill=True,
            fill_color="#EF9A9A",
            fill_opacity=0.7,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"Hot Spot: {lst_val:.1f}°C",
        ).add_to(hotspot_group)

    hotspot_group.add_to(m)

    # ── 3. Optimal Intervention Markers ─────────────────────────────────────
    if opt_allocation:
        intv_group = folium.FeatureGroup(name="Optimal Interventions", show=True)

        intv_colors = {
            "urban_greening": "#2E7D32",
            "cool_roofs":     "#CFD8DC",
            "green_roofs":    "#6abf69",
            "water_bodies":   "#4fc3f7",
            "permeable_pavements": "#BCAAA4",
            "street_trees":   "#388E3C",
        }
        intv_icons = {
            "urban_greening": "tree",
            "cool_roofs":     "home",
            "green_roofs":    "leaf",
            "water_bodies":   "tint",
            "permeable_pavements": "road",
            "street_trees":   "tree",
        }

        # Sample allocation map cells
        sampled_alloc = {k: v for k, v in list(opt_allocation.items())[::8]}

        for cell_idx, intv_name in sampled_alloc.items():
            if cell_idx >= len(lat_flat):
                continue
            lat_pt = float(lat_flat[cell_idx])
            lon_pt = float(lon_flat[cell_idx])
            color  = intv_colors.get(intv_name, "#4CAF50")

            popup_html = f"""
            <div style="font-family:Arial; min-width:180px;">
              <b style="color:{color};">INTERVENTION SITE</b><br>
              <hr style="margin:4px 0">
              <b>Type:</b> {intv_name.replace('_', ' ').title()}<br>
              <b>LST Before:</b> {float(lst_flat[cell_idx]):.1f}°C<br>
              <b>Coords:</b> {lat_pt:.4f}°N, {lon_pt:.4f}°E
            </div>"""

            folium.CircleMarker(
                location=[lat_pt, lon_pt],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=1.0,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{intv_name.replace('_', ' ').title()}",
            ).add_to(intv_group)

        intv_group.add_to(m)

    # ── 4. City Center & Legend ──────────────────────────────────────────────
    folium.Marker(
        location=center,
        popup=f"<b>{city['name']} City Center</b><br>{center[0]:.4f}°N, {center[1]:.4f}°E",
        tooltip="City Center",
        icon=folium.Icon(color="black", icon="info-sign"),
    ).add_to(m)

    # Legend HTML
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:10px;
                border:2px solid #ccc; font-family:Arial; font-size:12px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.2);">
      <b style="font-size:14px;">Urban Heat Map</b><br><hr style="margin:6px 0">
      <b>LST HeatMap:</b><br>
      <span style="color:#2196F3;">■</span> Cool
      <span style="color:#4CAF50;">■</span> Moderate
      <span style="color:#FF9800;">■</span> Warm
      <span style="color:#F44336;">■</span> Hot
      <span style="color:#7B1FA2;">■</span> Extreme<br><br>
      <b>Markers:</b><br>
      <span style="color:#c62828;">●</span> Significant Hot Spot (Gi*)<br>
      <span style="color:#2E7D32;">●</span> Urban Greening Site<br>
      <span style="color:#4fc3f7;">●</span> Water Body Site
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title ────────────────────────────────────────────────────────────────
    title_html = f"""
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                z-index:1000; background:rgba(13,71,161,0.92); color:white;
                padding:10px 24px; border-radius:10px; font-family:Arial;
                font-size:15px; font-weight:bold; text-align:center;
                box-shadow:2px 2px 8px rgba(0,0,0,0.4);">
      Urban Heat Mitigation — {city['name']} Interactive Map<br>
      <span style="font-size:11px; font-weight:normal; opacity:0.85;">
        LST | Hotspot Detection (Gi*) | Optimal Cooling Interventions
      </span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = os.path.join(output_dir, "interactive_heat_map.html")
    m.save(out_path)
    print(f"  [SAVE] Interactive map saved: {out_path}")
    return out_path
