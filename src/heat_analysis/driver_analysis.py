"""
driver_analysis.py
------------------
Quantifies the contribution of each urban heating driver using SHAP
(SHapley Additive exPlanations) values derived from the trained ML model.

Outputs:
  - Feature importance rankings
  - SHAP summary statistics per feature
  - Percentage contribution of each driver
"""

import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


FEATURE_LABELS = {
    "ndvi":                "Vegetation Index (NDVI)",
    "albedo":              "Surface Albedo",
    "svf":                 "Sky View Factor",
    "impervious_fraction": "Impervious Surface Fraction",
    "building_height":     "Building Height",
    "air_temp":            "Air Temperature (ERA5)",
    "humidity":            "Relative Humidity",
    "wind_speed":          "Wind Speed",
    "lulc":                "Land Use / Land Cover",
    "lulc_0":              "LULC: Water",
    "lulc_1":              "LULC: Dense Vegetation",
    "lulc_2":              "LULC: Sparse Vegetation",
    "lulc_3":              "LULC: Agriculture",
    "lulc_4":              "LULC: Low-density Residential",
    "lulc_5":              "LULC: High-density Residential",
    "lulc_6":              "LULC: Commercial/Industrial",
    "lulc_7":              "LULC: Barren/Construction",
    "lat":                 "Latitude",
    "lon":                 "Longitude",
    "is_water":            "Water Body Indicator",
}


def compute_shap_values(model, X_sample: pd.DataFrame,
                        model_type: str = "xgboost") -> tuple:
    """
    Compute SHAP values for a trained model.
    
    Parameters
    ----------
    model      : trained model (XGBoost, sklearn, or PyTorch)
    X_sample   : feature DataFrame (subset for efficiency)
    model_type : 'xgboost' or 'linear'
    
    Returns
    -------
    shap_values   : np.ndarray of shape (n_samples, n_features)
    explainer     : SHAP explainer object
    """
    print(f"  Computing SHAP values on {len(X_sample)} samples...")

    if model_type == "xgboost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Kernel SHAP for generic models (slower)
        background = shap.sample(X_sample, 100)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample, nsamples=200)

    print(f"  [OK] SHAP values computed. Shape: {shap_values.shape}")
    return shap_values, explainer


def get_driver_importance(shap_values: np.ndarray,
                          feature_names: list) -> pd.DataFrame:
    """
    Summarize SHAP-based driver importance.
    
    Returns DataFrame ranked by mean absolute SHAP value.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total = mean_abs_shap.sum()
    pct_contribution = (mean_abs_shap / total) * 100

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "label": [FEATURE_LABELS.get(f, f) for f in feature_names],
        "mean_abs_shap": mean_abs_shap,
        "pct_contribution": pct_contribution,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    return importance_df


def aggregate_driver_groups(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate LULC one-hot features and group drivers into categories.
    
    Categories:
      - Vegetation
      - Surface Properties
      - Urban Morphology
      - Atmospheric
      - Land Use
    """
    category_map = {
        "ndvi":                "Vegetation",
        "albedo":              "Surface Properties",
        "svf":                 "Urban Morphology",
        "building_height":     "Urban Morphology",
        "impervious_fraction": "Surface Properties",
        "air_temp":            "Atmospheric",
        "humidity":            "Atmospheric",
        "wind_speed":          "Atmospheric",
        "is_water":            "Vegetation",
        "lat":                 "Geographic",
        "lon":                 "Geographic",
    }

    rows = []
    for _, row in importance_df.iterrows():
        feat = row["feature"]
        if feat.startswith("lulc_"):
            cat = "Land Use / Land Cover"
        else:
            cat = category_map.get(feat, "Other")
        rows.append({"feature": feat, "label": row["label"],
                     "pct_contribution": row["pct_contribution"],
                     "category": cat})

    df = pd.DataFrame(rows)
    grouped = (df.groupby("category")["pct_contribution"]
               .sum()
               .reset_index()
               .sort_values("pct_contribution", ascending=False))
    return grouped


def analyze_drivers(model, X_train: pd.DataFrame,
                    feature_cols: list, n_shap_samples: int = 2000) -> dict:
    """
    Full driver analysis pipeline.
    
    Returns dict with importance_df, grouped_df, shap_values, feature_cols.
    """
    # Sample for SHAP efficiency
    sample_size = min(n_shap_samples, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)[feature_cols]

    shap_values, explainer = compute_shap_values(model, X_sample, model_type="xgboost")
    importance_df = get_driver_importance(shap_values, feature_cols)
    grouped_df = aggregate_driver_groups(importance_df)

    # Console printing removed to prevent Windows terminal encoding errors

    return {
        "importance_df": importance_df,
        "grouped_df": grouped_df,
        "shap_values": shap_values,
        "X_sample": X_sample,
        "feature_cols": feature_cols,
        "explainer": explainer,
    }

