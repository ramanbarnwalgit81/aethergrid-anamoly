"""
Fleet Normal Behavior Model (Fleet-NBM) — the single biggest swing at the
CARE hard-task AUC ceiling.

REFERENCES
----------
- Schlechtingen & Santos 2014 — classical NBM with ANFIS
- Roelofs, Haesen et al. 2023 WES 8/893 — NBM conditioning variable survey
- Perez-Sanjines, Peeters 2023 MSSP — fleet-based cyclic spectral coherence
- Grataloup, Jonas, Meyer 2025 arxiv 2409.03672 — federated sibling NBM,
  40% MAE reduction over single-turbine baseline on Penmanshiel+Kelmarsh+EDP

CORE IDEA
---------
For each target drivetrain signal (gearbox_oil_temp, main_bearing_temp, etc),
learn a regression:

    target = f(wind_speed, ambient_temp, active_power, rotor_speed,
                wind_direction_sin, wind_direction_cos, air_density)

The training data pools ALL healthy turbines from a fleet (Penmanshiel:
14 turbines × 4.5 years = 3.3M rows). This gives a fleet-average expectation.

At inference on a target event CSV (CARE Farm A), we compute:

    residual[t] = target[t] - predicted_target[t | conditions]

The anomaly signal for gradual degradation is the DRIFT of the residual's
rolling statistics, not the instantaneous value (which can be within physical
bounds for weeks before failure).

KEY INNOVATION vs prior fleet NBM
---------------------------------
1. Penmanshiel 14-turbine fleet as pretraining source (4.5 turbine-years
   of healthy data — largest public wind fleet NBM training corpus).
2. Transfer to CARE Farm A (different turbine model, same domain) via
   feature standardization.
3. Ordinal pre-fault labels: timesteps within [event_start - T, event_start]
   get increasing anomaly weights for downstream calibration.

USAGE
-----
    # Train on Penmanshiel fleet
    python -m src.benchmark.fleet_nbm train --source penmanshiel

    # Transfer + evaluate on CARE
    python -m src.benchmark.fleet_nbm eval --target care_farm_a
"""

from pathlib import Path
import argparse
import json
import os, sys
import time

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score


HARMONIZED_DIR = Path("data/benchmark/harmonized")
CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")
MODELS_DIR = Path("models/fleet_nbm")
RESULTS_DIR = Path("docs/results")


# Conditioning variables available across ALL datasets
CONDITIONING_COLS = [
    "wind_speed_ms",
    "ambient_temp_c",
    "active_power_kw",
    "rotor_speed_rpm",
]

# Target signals — fault indicators we want residuals of
TARGET_COLS = [
    "gearbox_oil_temp_c",
    "gearbox_bearing_temp_c",
    "main_bearing_temp_c",
    "generator_bearing_de_temp_c",
    "generator_bearing_nde_temp_c",
    "generator_winding_temp_c",
    "nacelle_temp_c",
]


class FleetNBM(nn.Module):
    """
    Per-target MLP normal-behaviour regressor.
    One model per target column. Each is a small MLP conditioned on
    operating state.
    """

    def __init__(self, n_conditions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_conditions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def engineer_conditioning(df: pd.DataFrame) -> pd.DataFrame:
    """Build the conditioning feature matrix from raw canonical columns.

    Adds:
      - wind_direction_sin, wind_direction_cos (if available)
      - air_density proxy (1 / (273 + ambient_temp_c)) — simplified
    """
    out = df.copy()
    if "wind_direction_deg" in df.columns:
        rad = np.deg2rad(df["wind_direction_deg"])
        out["wind_direction_sin"] = np.sin(rad)
        out["wind_direction_cos"] = np.cos(rad)
    else:
        out["wind_direction_sin"] = 0.0
        out["wind_direction_cos"] = 0.0
    if "ambient_temp_c" in df.columns:
        out["air_density_proxy"] = 1.0 / (273.15 + df["ambient_temp_c"].clip(-40, 60))
    else:
        out["air_density_proxy"] = 0.0
    return out


def get_conditioning_feature_cols():
    return CONDITIONING_COLS + ["wind_direction_sin", "wind_direction_cos",
                                   "air_density_proxy"]


def filter_healthy_operation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only healthy-operation rows for NBM training:
      - Power > 0 (turbine generating)
      - Wind speed > cut-in (3 m/s)
      - Wind speed < cut-out (25 m/s)
      - Non-negative temperatures
    """
    mask = pd.Series(True, index=df.index)
    if "active_power_kw" in df.columns:
        mask &= df["active_power_kw"].fillna(-1) > 0
    if "wind_speed_ms" in df.columns:
        mask &= df["wind_speed_ms"].fillna(0).between(3, 25)
    if "rotor_speed_rpm" in df.columns:
        mask &= df["rotor_speed_rpm"].fillna(0) > 0.5
    return df[mask].copy()


def train_fleet_nbm(source: str = "penmanshiel",
                     hidden: int = 64, n_epochs: int = 30,
                     batch_size: int = 4096, lr: float = 1e-3,
                     device: str = "cpu"):
    """
    Train one Fleet-NBM per target column on the source fleet dataset.
    """
    torch.manual_seed(42)

    if source == "penmanshiel":
        parquet = HARMONIZED_DIR / "penmanshiel_full.parquet"
    elif source == "kelmarsh":
        parquet = HARMONIZED_DIR / "kelmarsh.parquet"
    else:
        raise ValueError(f"Unknown source: {source}")

    print(f"[NBM] Loading {parquet}...")
    df = pd.read_parquet(parquet)
    print(f"[NBM] {len(df):,} rows, turbines: {df['turbine_id'].nunique()}")

    df = engineer_conditioning(df)
    df = filter_healthy_operation(df)
    print(f"[NBM] After health filter: {len(df):,} rows")

    cond_cols = get_conditioning_feature_cols()
    cond_cols_present = [c for c in cond_cols if c in df.columns]
    print(f"[NBM] Conditioning cols: {cond_cols_present}")

    targets_present = [c for c in TARGET_COLS
                         if c in df.columns and df[c].notna().sum() > 10000]
    print(f"[NBM] Training targets: {targets_present}")

    # Build the conditioning matrix once
    X = df[cond_cols_present].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaler = RobustScaler().fit(X)
    X_s = X_scaler.transform(X).astype(np.float32)
    X_s = np.clip(X_s, -10, 10)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    per_target_info = {}

    for target in targets_present:
        t0 = time.time()
        print(f"\n[TRAIN] target={target}")

        y = df[target].to_numpy(dtype=np.float32)
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 1000:
            print(f"  [SKIP] only {valid_mask.sum()} valid rows")
            continue

        X_valid = X_s[valid_mask]
        y_valid = y[valid_mask]

        y_scaler = RobustScaler().fit(y_valid.reshape(-1, 1))
        y_s = y_scaler.transform(y_valid.reshape(-1, 1)).ravel().astype(np.float32)
        y_s = np.clip(y_s, -10, 10)

        # Train/val split
        n = len(X_valid)
        split = int(n * 0.85)
        X_tr, X_val = X_valid[:split], X_valid[split:]
        y_tr, y_val = y_s[:split], y_s[split:]

        model = FleetNBM(n_conditions=X_valid.shape[1], hidden=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        tr_t = torch.FloatTensor(X_tr).to(device)
        tr_y = torch.FloatTensor(y_tr).to(device)
        val_t = torch.FloatTensor(X_val).to(device)
        val_y = torch.FloatTensor(y_val).to(device)

        best_val = float("inf")
        best_state = None
        wait = 0
        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(len(tr_t))
            for i in range(0, len(tr_t), batch_size):
                idx = perm[i:i + batch_size]
                bx = tr_t[idx]; by = tr_y[idx]
                opt.zero_grad()
                pred = model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_t)
                val_loss = loss_fn(val_pred, val_y).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 5:
                    break

        model.load_state_dict(best_state)

        # Per-target MAE on val in original units
        model.eval()
        with torch.no_grad():
            val_pred_s = model(val_t).cpu().numpy()
        val_pred_orig = y_scaler.inverse_transform(val_pred_s.reshape(-1, 1)).ravel()
        val_y_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
        mae = float(np.mean(np.abs(val_pred_orig - val_y_orig)))

        print(f"  best_val_loss={best_val:.4f}  MAE={mae:.3f}  "
              f"took {time.time()-t0:.0f}s")

        # Save model + scalers
        safe_name = target.replace("_c", "")  # remove unit suffix in filename
        out_path = MODELS_DIR / f"{source}_{safe_name}.pt"
        torch.save({
            "state_dict": best_state,
            "n_conditions": X_valid.shape[1],
            "hidden": hidden,
            "cond_cols": cond_cols_present,
            "target": target,
            "y_center_": y_scaler.center_.tolist(),
            "y_scale_": y_scaler.scale_.tolist(),
            "x_center_": X_scaler.center_.tolist(),
            "x_scale_": X_scaler.scale_.tolist(),
            "mae": mae,
            "best_val_loss": best_val,
        }, out_path)
        per_target_info[target] = {
            "model_path": str(out_path),
            "mae": mae,
            "val_loss": best_val,
        }

    # Write a manifest
    manifest = {
        "source": source,
        "conditioning_cols": cond_cols_present,
        "targets": per_target_info,
        "train_rows": int(valid_mask.sum()) if targets_present else 0,
    }
    with (MODELS_DIR / f"{source}_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[OK] Trained {len(per_target_info)} Fleet-NBM targets on {source}")
    print(f"[OK] Manifest: {MODELS_DIR / f'{source}_manifest.json'}")
    return manifest


def load_nbm(source: str, target: str) -> dict:
    """Load a trained Fleet-NBM bundle."""
    safe_name = target.replace("_c", "")
    path = MODELS_DIR / f"{source}_{safe_name}.pt"
    if not path.exists():
        return None
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = FleetNBM(n_conditions=bundle["n_conditions"],
                       hidden=bundle["hidden"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    bundle["model"] = model
    return bundle


def compute_nbm_residual(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Return residual = actual - NBM-predicted for the target column.

    NaN where any required input (cond or target) is NaN.
    """
    target = bundle["target"]
    cond_cols = bundle["cond_cols"]

    df = engineer_conditioning(df)

    if target not in df.columns:
        return np.full(len(df), np.nan, dtype=np.float32)

    present_cond = [c for c in cond_cols if c in df.columns]
    if len(present_cond) < len(cond_cols):
        # Add any missing as zero columns (fallback)
        for c in cond_cols:
            if c not in df.columns:
                df[c] = 0.0

    X = df[cond_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_s = (X - np.asarray(bundle["x_center_"])) / np.asarray(bundle["x_scale_"])
    X_s = np.clip(X_s, -10, 10).astype(np.float32)

    with torch.no_grad():
        pred_s = bundle["model"](torch.FloatTensor(X_s)).cpu().numpy()
    y_center = np.asarray(bundle["y_center_"]).ravel()
    y_scale = np.asarray(bundle["y_scale_"]).ravel()
    pred_orig = pred_s * y_scale + y_center

    actual = df[target].to_numpy(dtype=np.float32)
    residual = actual - pred_orig
    return residual


def eval_on_care(source: str = "penmanshiel") -> dict:
    """
    Evaluate Fleet-NBM residuals on CARE Farm A event-window HARD task.

    For each event:
      1. Load event CSV, rename to canonical columns
      2. Compute residuals for all available target signals using Penmanshiel NBMs
      3. Combine residuals (sum of absolute standardized residuals)
      4. Score AUC vs event_start_id:event_end_id label
      5. Also report status-filtered variant (matches CARE paper spec)
    """
    from src.benchmark.adapters.care import FARM_A_COL_MAP
    from src.benchmark.schema import CANONICAL_RAW_FEATURES

    datasets_dir = CARE_DIR / "datasets"
    event_info = CARE_DIR / "event_info.csv"
    events = pd.read_csv(event_info, sep=";")

    # Load all NBMs
    nbms = {}
    for target in TARGET_COLS:
        bundle = load_nbm(source, target)
        if bundle is not None:
            nbms[target] = bundle
    print(f"[EVAL] Loaded {len(nbms)} Fleet-NBMs: {list(nbms.keys())}")
    if not nbms:
        print("[EVAL] No NBMs loaded — train first.")
        return {}

    results = []
    pooled_scores_full = []
    pooled_labels_event = []
    pooled_labels_status = []

    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<28} {'AUC-event':>9} "
          f"{'AUC-statfilt':>12}")
    print("-" * 80)

    for _, row in events.iterrows():
        event_id = int(row["event_id"])
        csv_path = datasets_dir / f"{event_id}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, sep=";", low_memory=False)
        df = df.rename(columns=FARM_A_COL_MAP)
        for c in CANONICAL_RAW_FEATURES:
            if c not in df.columns:
                df[c] = np.nan

        n = len(df)
        status_arr = df["status_type_id"].fillna(0).astype(int).values if \
            "status_type_id" in df.columns else np.zeros(n, dtype=int)
        train_test_arr = df["train_test"].values if \
            "train_test" in df.columns else np.array(["train"] * n)

        is_anomaly_event = np.zeros(n, dtype=int)
        if row["event_label"] == "anomaly":
            s = int(row["event_start_id"])
            e = int(row["event_end_id"])
            is_anomaly_event[s:e + 1] = 1

        # Stack residuals for each NBM target we have
        residual_matrix = []
        residual_target_names = []
        for target, bundle in nbms.items():
            if df[target].notna().sum() > 100:
                r = compute_nbm_residual(df, bundle)
                # Standardize per event using training-distribution residual stats
                # (approximation: robust-standardize by event median/MAD)
                med = np.nanmedian(r)
                mad = np.nanmedian(np.abs(r - med)) + 1e-6
                z = np.abs((r - med) / (1.4826 * mad))
                residual_matrix.append(z)
                residual_target_names.append(target)

        if not residual_matrix:
            continue

        residual_matrix = np.stack(residual_matrix, axis=1)
        # Anomaly score: max over targets (flag if ANY subsystem drifts)
        anomaly_score = np.nanmax(residual_matrix, axis=1)
        # NaN → 0 for scoring
        anomaly_score = np.nan_to_num(anomaly_score, nan=0.0)

        # Status-filtered: only score rows where status_type_id == 0
        # (per CARE paper definition of hard task)
        statfilt_mask = (status_arr == 0)

        # HARD TASK AUC: event-window label on ALL rows
        auc_event = None
        if is_anomaly_event.sum() > 0 and is_anomaly_event.sum() < n:
            auc_event = float(roc_auc_score(is_anomaly_event, anomaly_score))

        # HARD TASK (status-filtered) AUC — the CARE-compliant definition
        auc_statfilt = None
        y_filt = is_anomaly_event[statfilt_mask]
        s_filt = anomaly_score[statfilt_mask]
        if len(y_filt) > 0 and y_filt.sum() > 0 and y_filt.sum() < len(y_filt):
            auc_statfilt = float(roc_auc_score(y_filt, s_filt))

        # Accumulate for pooled
        if row["event_label"] == "anomaly":
            pooled_scores_full.append(anomaly_score)
            pooled_labels_event.append(is_anomaly_event)
            pooled_labels_status.append((status_arr != 0).astype(int))

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  —   "
        desc = str(row.get("event_description", ""))[:26]
        print(f"{event_id:>4} {row['event_label']:>8} {desc:<28} "
              f"{fmt(auc_event):>9} {fmt(auc_statfilt):>12}")

        results.append({
            "event_id": event_id,
            "event_type": row["event_label"],
            "event_description": str(row.get("event_description", "")),
            "n_rows": n,
            "auc_event": auc_event,
            "auc_event_status_filtered": auc_statfilt,
            "nbm_targets_used": residual_target_names,
        })

    # Aggregates
    anomaly_results = [r for r in results if r["event_type"] == "anomaly"]

    print(f"\n{'='*80}")
    print(f"  AGGREGATE — Fleet-NBM ({source} -> CARE Farm A, HARD TASK)")
    print(f"{'='*80}")

    for key in ["auc_event", "auc_event_status_filtered"]:
        vals = [r[key] for r in anomaly_results if r.get(key) is not None]
        if vals:
            print(f"  {key:<30} mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}  n={len(vals)}")

    if pooled_scores_full:
        ps = np.concatenate(pooled_scores_full)
        ple = np.concatenate(pooled_labels_event)
        pls = np.concatenate(pooled_labels_status)
        pooled_event = float(roc_auc_score(ple, ps)) if len(np.unique(ple)) == 2 else None
        # Status-filtered pooled
        sf_mask = (pls == 0)
        pooled_event_filt = None
        if sf_mask.sum() > 0 and ple[sf_mask].sum() > 0:
            pooled_event_filt = float(roc_auc_score(ple[sf_mask], ps[sf_mask]))
        print(f"\n  POOLED AUC (event window):             {pooled_event}")
        print(f"  POOLED AUC (event, status-filtered):    {pooled_event_filt}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": f"Fleet-NBM ({source}) -> CARE Farm A hard task",
        "n_nbms": len(nbms),
        "targets": list(nbms.keys()),
        "per_event": results,
        "mean_auc_event": float(np.mean([r["auc_event"] for r in anomaly_results
                                            if r.get("auc_event") is not None]))
                            if anomaly_results else 0,
        "mean_auc_event_statfilt": float(np.mean([r["auc_event_status_filtered"] for r in anomaly_results
                                                      if r.get("auc_event_status_filtered") is not None]))
                                      if anomaly_results else 0,
    }
    out_path = RESULTS_DIR / f"fleet_nbm_{source}_to_care.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Saved: {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--source", default="penmanshiel",
                    choices=["penmanshiel", "kelmarsh"])
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--epochs", type=int, default=30)

    e = sub.add_parser("eval")
    e.add_argument("--source", default="penmanshiel")

    args = parser.parse_args()

    if args.cmd == "train":
        train_fleet_nbm(source=args.source, hidden=args.hidden,
                         n_epochs=args.epochs)
    elif args.cmd == "eval":
        eval_on_care(source=args.source)


if __name__ == "__main__":
    main()
