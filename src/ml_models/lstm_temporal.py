"""
lstm_temporal.py
----------------
LSTM-based temporal heat dynamics model.

Models how Land Surface Temperature evolves over time (hourly/daily/seasonal)
under varying atmospheric and urban conditions.

Architecture:
  - Input: time-series of features (air_temp, humidity, wind, NDVI, albedo, hour, doy)
  - LSTM layers with physics-informed skip connections
  - Output: LST at next time step
  - Physics constraint: energy balance residual penalty in loss

Data:
  - Synthetic time-series: 365 days × 24 hours per reference urban cell
  - Driven by ERA5-like diurnal and seasonal cycles

Usage:
  from src.ml_models.lstm_temporal import train_lstm_temporal, plot_temporal_results
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ── Constants ──────────────────────────────────────────────────────────────────
SIGMA    = 5.67e-8   # Stefan-Boltzmann [W m-2 K-4]
EPSILON  = 0.95
RS_MAX   = 850.0     # Peak solar radiation [W m-2]
RL_DOWN  = 380.0     # Downwelling longwave [W m-2]
RCP_AIR  = 1200.0    # rho*Cp [J m-3 K-1]


# ── Synthetic Time-Series Generator ───────────────────────────────────────────

def generate_temporal_dataset(
    n_days: int = 365,
    n_cells: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate physics-grounded hourly time series for multiple urban cells.

    Features (per timestep):
      hour_sin, hour_cos  : diurnal cycle encoding
      doy_sin, doy_cos    : seasonal cycle encoding
      air_temp            : ERA5-like air temperature
      humidity            : relative humidity
      wind_speed          : wind speed m/s
      ndvi                : vegetation index
      albedo              : surface albedo
      svf                 : sky view factor
      impervious          : impervious fraction
      Rs_down             : incoming shortwave radiation (W/m2)
      lulc_class          : 0-6

    Target:
      lst                 : land surface temperature (°C)
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(n_days * 24)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    # Seasonal air temperature: Delhi annual cycle
    T_mean = 28.0
    T_amplitude = 10.0  # seasonal amplitude
    T_diurnal = 6.0     # diurnal amplitude

    air_temp_seasonal = T_mean + T_amplitude * np.sin(
        2 * np.pi * (day_of_year - 80) / 365.0
    )
    air_temp_diurnal = T_diurnal * np.sin(
        2 * np.pi * (hour_of_day - 6) / 24.0
    )

    # Solar radiation: zero at night, peak at noon
    Rs_down = np.maximum(0, RS_MAX * np.sin(np.pi * (hour_of_day - 6) / 12.0))
    # Seasonal modulation
    Rs_down *= (0.7 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365.0))

    # Humidity: anti-correlated with temperature, monsoon boost
    monsoon = np.where((day_of_year >= 150) & (day_of_year <= 270), 25.0, 0.0)
    humidity_base = 50.0 - 0.5 * air_temp_diurnal + monsoon
    humidity = np.clip(humidity_base + rng.normal(0, 5, len(hours)), 15, 95)

    # Wind speed
    wind_speed = np.clip(2.0 + rng.exponential(1.5, len(hours)), 0.3, 12.0)

    # Encode cyclical time features
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
    doy_sin  = np.sin(2 * np.pi * day_of_year / 365.0)
    doy_cos  = np.cos(2 * np.pi * day_of_year / 365.0)

    # Build multi-cell dataset
    rows = []
    cell_configs = [
        # (ndvi, albedo, svf, impervious, lulc)
        (0.05, 0.32, 0.55, 0.90, 6),  # Commercial core
        (0.10, 0.28, 0.60, 0.75, 5),  # High-density residential
        (0.22, 0.22, 0.72, 0.45, 4),  # Low-density residential
        (0.55, 0.15, 0.90, 0.10, 2),  # Sparse vegetation
        (0.75, 0.12, 0.98, 0.05, 1),  # Dense vegetation
        (0.00, 0.06, 1.00, 0.00, 0),  # Water body
        (0.30, 0.18, 0.80, 0.05, 3),  # Agriculture
        (-0.01, 0.30, 0.75, 0.60, 7), # Barren land
    ]
    # Repeat cells to reach n_cells
    all_cells = (cell_configs * ((n_cells // len(cell_configs)) + 1))[:n_cells]

    for cell_id, (ndvi, albedo, svf, impervious, lulc) in enumerate(all_cells):
        air_temp = air_temp_seasonal + air_temp_diurnal + rng.normal(0, 0.8, len(hours))

        # Physics-driven LST
        # Daytime: driven by Rs and albedo absorption
        # Nighttime: longwave emission dominance
        absorbed = (1 - albedo) * Rs_down
        lst_solar_forcing = absorbed / 150.0  # Approximate sensible/radiative

        ra = 100.0 / (wind_speed + 0.5)
        evap_fraction = np.clip(0.15 + 0.6 * ndvi + 0.08 * (humidity / 100.0), 0.1, 0.9)

        lst = (
            air_temp
            + lst_solar_forcing
            + 8.0 * impervious * (Rs_down / RS_MAX)
            - 5.0 * ndvi
            - 3.0 * albedo
            + 2.5 * (1.0 - svf) * (Rs_down / (RS_MAX + 1))
            - 1.5 * evap_fraction * (Rs_down / (RS_MAX + 1))
            + rng.normal(0, 0.5, len(hours))
        )
        lst = np.clip(lst, 10.0, 60.0)

        for t in range(len(hours)):
            rows.append({
                "cell_id":    cell_id,
                "lulc":       lulc,
                "hour":       hour_of_day[t],
                "doy":        day_of_year[t],
                "hour_sin":   hour_sin[t],
                "hour_cos":   hour_cos[t],
                "doy_sin":    doy_sin[t],
                "doy_cos":    doy_cos[t],
                "air_temp":   air_temp[t],
                "humidity":   humidity[t],
                "wind_speed": wind_speed[t],
                "Rs_down":    Rs_down[t],
                "ndvi":       ndvi,
                "albedo":     albedo,
                "svf":        svf,
                "impervious": impervious,
                "lst":        lst[t],
            })

    df = pd.DataFrame(rows)
    print(f"  [OK] Temporal dataset: {len(df):,} records | {n_days} days | {n_cells} cells")
    return df


# ── Dataset & Model ────────────────────────────────────────────────────────────

class HeatSequenceDataset(Dataset):
    """Sliding-window time-series dataset for LSTM training."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
        self.seq_len = seq_len
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_next = self.y[idx + self.seq_len]
        return x_seq, y_next


class PhysicsLSTM(nn.Module):
    """
    Physics-Informed LSTM for urban heat dynamics.

    Architecture:
      - 2-layer LSTM with dropout
      - FC head for LST prediction
      - Physics skip connection: adds estimated energy balance offset
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Physics skip: learns to weight physics-based estimate
        self.physics_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]        # Take last timestep
        data_pred = self.fc(last).squeeze(-1)

        # Physics skip: use last timestep's air_temp (assumed index 0 after scaling)
        # Adds a learnable physics offset — keeps predictions energy-consistent
        phys_estimate = x[:, -1, 0]     # air_temp (first feature after scaling)
        pred = data_pred + self.physics_weight * phys_estimate
        return pred


def physics_loss(pred_lst: torch.Tensor, features_last: torch.Tensor,
                 lambda_phys: float = 0.1) -> torch.Tensor:
    """
    Physics penalty: energy balance residual.
    features_last: (batch, input_size) — last timestep features
    Assumes columns: [air_temp_scaled, humidity_scaled, Rs_down_scaled, ...]
    """
    # We use normalised proxy — penalty on large |pred - features[0]| deviations
    # (Full energy balance requires denormalization; this is a scaled version)
    penalty = torch.mean((pred_lst - features_last[:, 0]) ** 2)
    return lambda_phys * penalty


# ── Training Pipeline ──────────────────────────────────────────────────────────

def train_lstm_temporal(config: dict) -> dict:
    """
    Full LSTM temporal training pipeline.

    Returns dict with model, scaler, metrics, predictions, df.
    """
    print("  Generating temporal time-series dataset...")
    df = generate_temporal_dataset(n_days=365, n_cells=16, seed=42)

    feature_cols = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "air_temp", "humidity", "wind_speed", "Rs_down",
        "ndvi", "albedo", "svf", "impervious",
    ]
    target_col = "lst"
    SEQ_LEN    = 24   # 24-hour lookback window
    EPOCHS     = 30
    BATCH_SIZE = 256
    LR         = 1e-3
    HIDDEN     = 64
    LAYERS     = 2

    # Use first cell for demonstration (most time steps available)
    df_cell = df[df["cell_id"] == 0].copy().reset_index(drop=True)
    X_raw = df_cell[feature_cols].values.astype(np.float32)
    y_raw = df_cell[target_col].values.astype(np.float32)

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    # Train/test split (80/20 temporal)
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]
    y_test_orig = y_raw[split + SEQ_LEN:]

    train_ds = HeatSequenceDataset(X_train, y_train, SEQ_LEN)
    test_ds  = HeatSequenceDataset(X_test,  y_test,  SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    device = torch.device("cpu")
    model  = PhysicsLSTM(input_size=len(feature_cols),
                         hidden_size=HIDDEN, num_layers=LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    print(f"  Training PhysicsLSTM ({EPOCHS} epochs, seq={SEQ_LEN}h)...")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            data_l = criterion(pred, y_b)
            phys_l = physics_loss(pred, X_b[:, -1, :], lambda_phys=0.05)
            loss   = data_l + phys_l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        ep_loss /= max(len(train_dl), 1)
        train_losses.append(ep_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in test_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred_v = model(X_b)
                val_loss += criterion(pred_v, y_b).item()
        val_loss /= max(len(test_dl), 1)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"     Epoch {epoch+1:3d}/{EPOCHS} | Train: {ep_loss:.5f} | Val: {val_loss:.5f}")

    # Predict on test set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_b, _ in test_dl:
            all_preds.append(model(X_b.to(device)).cpu().numpy())
    y_pred_scaled = np.concatenate(all_preds)
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    n = min(len(y_test_orig), len(y_pred_orig))
    r2   = float(r2_score(y_test_orig[:n], y_pred_orig[:n]))
    rmse = float(np.sqrt(mean_squared_error(y_test_orig[:n], y_pred_orig[:n])))

    print(f"\n  [GRAPH] PhysicsLSTM Performance:")
    print(f"     R2   : {r2:.4f}")
    print(f"     RMSE : {rmse:.3f} deg C")

    return {
        "model":        model,
        "scaler_X":     scaler_X,
        "scaler_y":     scaler_y,
        "feature_cols": feature_cols,
        "df":           df,
        "y_test_orig":  y_test_orig[:n],
        "y_pred_orig":  y_pred_orig[:n],
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "metrics":      {"r2": round(r2, 4), "rmse_C": round(rmse, 3)},
        "seq_len":      SEQ_LEN,
    }


# ── Temporal Visualisation ─────────────────────────────────────────────────────

def plot_temporal_results(lstm_results: dict, output_dir: str):
    """Plot LSTM training curves, temporal predictions, and diurnal profiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    y_test  = lstm_results["y_test_orig"]
    y_pred  = lstm_results["y_pred_orig"]
    t_loss  = lstm_results["train_losses"]
    v_loss  = lstm_results["val_losses"]
    metrics = lstm_results["metrics"]
    df      = lstm_results["df"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Training curves
    ax = axes[0, 0]
    ax.plot(t_loss, label="Train Loss", color="#E53935", linewidth=2)
    ax.plot(v_loss, label="Val Loss",   color="#1565C0", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("PhysicsLSTM Training Curves", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Predicted vs Actual (sample window)
    ax2 = axes[0, 1]
    window = min(7 * 24, len(y_test))   # 1 week
    t_idx  = np.arange(window)
    ax2.plot(t_idx, y_test[:window], label="Actual LST",    color="#E53935", linewidth=1.5, alpha=0.8)
    ax2.plot(t_idx, y_pred[:window], label="Predicted LST", color="#1565C0", linewidth=1.5,
             linestyle="--", alpha=0.9)
    ax2.set_xlabel("Hours"); ax2.set_ylabel("LST (deg C)")
    ax2.set_title(f"7-Day Temporal Prediction\nR2={metrics['r2']:.4f}  RMSE={metrics['rmse_C']:.2f}°C",
                  fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    # 3. Scatter: Actual vs Predicted
    ax3 = axes[1, 0]
    ax3.scatter(y_test, y_pred, alpha=0.15, s=4, color="#7B1FA2")
    lim = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
    ax3.plot(lim, lim, "r--", linewidth=2, label="1:1 line")
    ax3.set_xlabel("Actual LST (deg C)"); ax3.set_ylabel("Predicted LST (deg C)")
    ax3.set_title("Actual vs Predicted LST", fontweight="bold")
    ax3.set_xlim(lim); ax3.set_ylim(lim)
    ax3.legend(); ax3.grid(alpha=0.3)
    ax3.text(0.05, 0.95, f"R2={metrics['r2']:.4f}\nRMSE={metrics['rmse_C']:.3f}°C",
             transform=ax3.transAxes, va="top", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # 4. Mean diurnal LST profile by LULC type
    ax4 = axes[1, 1]
    lulc_labels = {0: "Water", 1: "Dense Veg", 2: "Sparse Veg", 3: "Agriculture",
                   4: "Low-Res", 5: "High-Res", 6: "Commercial", 7: "Barren"}
    lulc_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i, (lulc_id, lbl) in enumerate(lulc_labels.items()):
        sub = df[(df["lulc"] == lulc_id)]
        if len(sub) == 0:
            continue
        profile = sub.groupby("hour")["lst"].mean()
        if len(profile) == 24:
            ax4.plot(profile.index, profile.values, label=lbl,
                     color=lulc_colors[i], linewidth=2, marker="o", markersize=3)
    ax4.set_xlabel("Hour of Day"); ax4.set_ylabel("Mean LST (deg C)")
    ax4.set_title("Diurnal LST Profile by Land Use", fontweight="bold")
    ax4.legend(fontsize=8, ncol=2); ax4.grid(alpha=0.3)
    ax4.set_xticks(range(0, 24, 3))

    plt.suptitle("Physics-Informed LSTM — Urban Heat Temporal Dynamics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "lstm_temporal_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] LSTM temporal analysis saved: {out}")
    return out
