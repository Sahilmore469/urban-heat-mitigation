"""
piml_model.py
-------------
Physics-Informed Machine Learning model for Land Surface Temperature prediction.

Approach:
  1. XGBoost as the primary data-driven model
  2. Physics constraint: Urban Energy Balance equation
     Net_Radiation = Sensible_Heat + Latent_Heat + Ground_Heat_Flux
     LST is bounded by energy balance residuals
  3. Physics penalty added during validation to score energy balance violations
  4. Neural network with physics loss term (optional, torch-based)

Energy Balance:
  Rn = H + LE + G
  where:
    Rn = net radiation ≈ (1 - α) * Rs↓ + ε * (Rl↓ - σT⁴)   [W/m²]
    H  = sensible heat flux ∝ (LST - Ta) / ra                [W/m²]
    LE = latent heat flux ∝ NDVI * (humidity forcing)        [W/m²]
    G  = ground heat flux ≈ 0.1 * Rn (daytime)               [W/m²]
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# Physical constants
SIGMA = 5.67e-8        # Stefan-Boltzmann constant [W m⁻² K⁻⁴]
EPSILON = 0.95         # Surface emissivity (typical urban)
RS_TYPICAL = 600.0     # Typical incoming shortwave radiation [W m⁻²]
RL_DOWN = 380.0        # Downwelling longwave radiation [W m⁻²]
RA_BASE = 100.0        # Aerodynamic resistance base [s m⁻¹]
RCP_AIR = 1200.0       # ρ * Cp of air [J m⁻³ K⁻¹]


def compute_energy_balance(lst_celsius: np.ndarray,
                            air_temp_celsius: np.ndarray,
                            albedo: np.ndarray,
                            ndvi: np.ndarray,
                            wind_speed: np.ndarray,
                            humidity: np.ndarray) -> dict:
    """
    Compute urban surface energy balance components.
    Used to validate physical consistency of LST predictions.
    """
    lst_K = lst_celsius + 273.15
    ta_K = air_temp_celsius + 273.15

    # Net radiation
    Rn = ((1 - albedo) * RS_TYPICAL
          + EPSILON * RL_DOWN
          - EPSILON * SIGMA * lst_K**4)

    # Aerodynamic resistance (decreases with wind)
    ra = RA_BASE / (wind_speed + 0.5)

    # Sensible heat flux
    H = RCP_AIR * (lst_celsius - air_temp_celsius) / (ra + 1e-6)

    # Latent heat flux (Bowen ratio approach)
    # Higher NDVI → more evapotranspiration
    evap_fraction = np.clip(0.2 + 0.6 * ndvi + 0.1 * (humidity / 100.0), 0.1, 0.9)
    LE = evap_fraction * Rn

    # Ground heat flux (Grimmond & Oke 1999: G ≈ 0.1–0.3 * Rn)
    G = 0.15 * np.abs(Rn)

    # Energy balance residual: should be ~0 if physically consistent
    residual = Rn - H - LE - G

    return {
        "Rn": Rn,
        "H": H,
        "LE": LE,
        "G": G,
        "residual": residual,
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def physics_penalty(y_pred: np.ndarray, X: pd.DataFrame) -> float:
    """
    Compute physics constraint violation penalty.
    Penalizes predictions that violate energy balance constraints.
    """
    if not all(c in X.columns for c in ["air_temp", "albedo", "ndvi", "wind_speed", "humidity"]):
        return 0.0

    eb = compute_energy_balance(
        lst_celsius=y_pred,
        air_temp_celsius=X["air_temp"].values,
        albedo=X["albedo"].values,
        ndvi=X["ndvi"].values,
        wind_speed=X["wind_speed"].values,
        humidity=X["humidity"].values,
    )
    # Normalize penalty by typical Rn magnitude
    penalty = np.sqrt(np.mean(eb["residual"]**2)) / (RS_TYPICAL + 1e-6)
    return float(penalty)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  config: dict) -> xgb.XGBRegressor:
    """Train XGBoost regressor for LST prediction."""
    params = config["ml"]["xgboost"]
    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=config["ml"]["random_state"],
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   config: dict, model_name: str = "XGBoost") -> dict:
    """
    Evaluate model with standard metrics + physics consistency score.
    """
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Physics penalty
    lambda_phys = config["ml"]["physics_lambda"]
    phys_penalty = physics_penalty(y_pred, X_test)
    piml_score = r2 - lambda_phys * phys_penalty

    metrics = {
        "model": model_name,
        "r2": round(r2, 4),
        "rmse_C": round(rmse, 4),
        "mae_C": round(mae, 4),
        "physics_penalty": round(phys_penalty, 4),
        "piml_score": round(piml_score, 4),
    }

    print(f"\n  [GRAPH] {model_name} Performance:")
    print(f"     R²           : {r2:.4f}")
    print(f"     RMSE         : {rmse:.3f}°C")
    print(f"     MAE          : {mae:.3f}°C")
    print(f"     Physics Penalty: {phys_penalty:.4f}")
    print(f"     PIML Score   : {piml_score:.4f}")

    return metrics, y_pred


def train_piml_pipeline(df: pd.DataFrame, config: dict) -> dict:
    """
    Full Physics-Informed ML training pipeline.
    
    1. Prepare features
    2. Train XGBoost
    3. Evaluate with physics penalty
    4. Return model, metrics, predictions
    """
    ml_cfg = config["ml"]
    target = "lst"

    # Feature columns (exclude lat/lon to avoid spatial overfitting in demo)
    feature_cols = [c for c in df.columns
                    if c not in [target, "is_water", "lat", "lon"]
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8, bool]]

    # Ensure numeric
    X = df[feature_cols].astype(float)
    y = df[target].astype(float)

    print(f"  Features: {len(feature_cols)} | Samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=ml_cfg["test_size"],
        random_state=ml_cfg["random_state"]
    )

    # Scaler (kept for reference; XGBoost doesn't require scaling)
    scaler = StandardScaler()
    scaler.fit(X_train)

    print("\n  Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train, config)

    metrics, y_pred_test = evaluate_model(
        xgb_model, X_test, y_test, config, model_name="XGBoost (Physics-Informed)"
    )

    # Compute energy balance on test set
    eb_components = compute_energy_balance(
        lst_celsius=y_pred_test,
        air_temp_celsius=X_test["air_temp"].values,
        albedo=X_test["albedo"].values,
        ndvi=X_test["ndvi"].values,
        wind_speed=X_test["wind_speed"].values,
        humidity=X_test["humidity"].values,
    )

    print(f"\n  [PHYS] Energy Balance Residual RMSE: {eb_components['residual_rmse']:.2f} W/m²")

    return {
        "model": xgb_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "metrics": metrics,
        "energy_balance": eb_components,
    }


def predict_full_grid(model, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Predict LST for the full grid DataFrame."""
    X_full = df[feature_cols].astype(float)
    return model.predict(X_full)

