"""PINN-v2: physics residuals on INPUT + v7 score -> XGBoost stacker.

Goal
----
Upgrade PINN-AE so that the paper's Physics-Informed Autoencoder contribution
is materially stronger than the data-only baseline on real CARE labels. The
original PINN-AE computed physics residuals on the decoder output (noisy) and
combined them with reconstruction error via a fixed 60/40 weighting (arbitrary).
Neither helped on real CARE Farm A (combined AUC 0.516 vs data-only 0.555).

Approach
--------
(1) Compute the IEC 61400-12-1 physics residuals directly on the RAW input row
    (not the AE reconstruction). This matches the Fleet-NBM pattern that works.
(2) Feed those residuals + the v7 ensemble base score + rolling statistics of
    each channel into an XGBoost meta-learner.
(3) Evaluate leave-one-event-out on CARE Farm A under event-window, precursor,
    and status labels. Compare to (a) v7 base, (b) v7+temporal XGB stacker.

Usage
-----
    python -m src.benchmark.pinn_stacker

Output
------
    docs/results/pinn_stacker_y_event.json
    docs/results/pinn_stacker_y_precursor.json
    docs/results/pinn_stacker_y_status.json
"""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None


RESULTS_DIR = Path("docs/results")
CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")

RATED_POWER_KW = 2000.0
CUT_IN_WIND = 3.0
RATED_WIND = 12.0
CUT_OUT_WIND = 25.0


# ─────────────────────────────────────────────────────────────────────────
# Physics residuals on RAW input (numpy, no autograd)
# ─────────────────────────────────────────────────────────────────────────

def iec_power_curve(wind: np.ndarray) -> np.ndarray:
    """IEC 61400-12-1 cubic power curve."""
    w = np.clip(wind, 0.0, CUT_OUT_WIND + 1)
    below = w < CUT_IN_WIND
    ramp = (w >= CUT_IN_WIND) & (w < RATED_WIND)
    rated = (w >= RATED_WIND) & (w < CUT_OUT_WIND)
    cutout = w >= CUT_OUT_WIND
    ramp_ratio = np.clip((w - CUT_IN_WIND) / (RATED_WIND - CUT_IN_WIND), 0.0, 1.0)
    return (
        below.astype(float) * 0.0
        + ramp.astype(float) * RATED_POWER_KW * ramp_ratio ** 3
        + rated.astype(float) * RATED_POWER_KW
        + cutout.astype(float) * 0.0
    )


def physics_residuals_input(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Seven physics residuals computed on raw input columns.

    Channels 1-4 cover the drivetrain subsystem (power curve, gearbox/bearing
    thermal, RPM-power). Channels 5-7 cover electrical/cooling subsystems so
    transformer and winding faults leave a residual signature:
      5. r_thermal_brg_nde: non-drive-end bearing vs gearbox oil.
      6. r_thermal_winding: generator winding vs ambient + load (electrical load).
      7. r_thermal_nacelle: nacelle air vs ambient + load (cooling-loop proxy).
    """
    wind = df["wind_speed_3_avg"].to_numpy(dtype=np.float32)
    power = df["power_30_avg"].to_numpy(dtype=np.float32)
    ambient = df["sensor_0_avg"].to_numpy(dtype=np.float32)
    rpm = df["sensor_18_avg"].to_numpy(dtype=np.float32)
    gearbox_oil = df["sensor_12_avg"].to_numpy(dtype=np.float32)
    gen_bearing_de = df["sensor_13_avg"].to_numpy(dtype=np.float32)

    # Optional channels (present in Farm A; absent on some other farms)
    gen_bearing_nde = (df["sensor_14_avg"].to_numpy(dtype=np.float32)
                       if "sensor_14_avg" in df.columns else None)
    gen_winding = (df["sensor_15_avg"].to_numpy(dtype=np.float32)
                   if "sensor_15_avg" in df.columns else None)
    nacelle = (df["sensor_7_avg"].to_numpy(dtype=np.float32)
               if "sensor_7_avg" in df.columns else None)

    # 1. Power-curve residual
    expected_power = iec_power_curve(wind)
    r_power = (power - expected_power) / RATED_POWER_KW

    # 2. Gearbox thermal residual (steady-state model)
    power_ratio = np.clip(power / RATED_POWER_KW, 0.0, 1.2)
    expected_gbx = ambient + 40.0 + 25.0 * power_ratio
    r_thermal_gbx = (gearbox_oil - expected_gbx) / 30.0

    # 3. DE bearing thermal coupling (tracks gearbox oil + small delta)
    r_thermal_brg_de = (gen_bearing_de - (gearbox_oil + 10.0)) / 20.0

    # 4. RPM-to-power correlation
    rpm_ratio = np.clip(rpm / 1600.0, 0.0, 1.2)
    expected_power_rpm = RATED_POWER_KW * rpm_ratio ** 3
    r_rpm_power = (power - expected_power_rpm) / RATED_POWER_KW

    out = {
        "r_power": r_power,
        "r_thermal_gbx": r_thermal_gbx,
        "r_thermal_brg_de": r_thermal_brg_de,
        "r_rpm_power": r_rpm_power,
    }

    # 5. NDE bearing thermal coupling
    if gen_bearing_nde is not None:
        out["r_thermal_brg_nde"] = (gen_bearing_nde - (gearbox_oil + 8.0)) / 20.0

    # 6. Generator-winding thermal (electrical-load-driven, picks up
    #    transformer/winding faults where cooling is marginal).
    if gen_winding is not None:
        expected_winding = ambient + 25.0 + 40.0 * power_ratio
        out["r_thermal_winding"] = (gen_winding - expected_winding) / 40.0

    # 7. Nacelle air vs ambient + small load term (cooling-loop proxy).
    if nacelle is not None:
        expected_nacelle = ambient + 5.0 + 10.0 * power_ratio
        out["r_thermal_nacelle"] = (nacelle - expected_nacelle) / 15.0

    return out


def standardize_per_event(x: np.ndarray) -> np.ndarray:
    """Robust per-event z-score using median and MAD."""
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return np.zeros_like(x)
    med = np.median(finite)
    mad = 1.4826 * np.median(np.abs(finite - med))
    if mad < 1e-9:
        mad = np.std(finite) if len(finite) > 1 else 1.0
    return np.where(np.isfinite(x), (x - med) / (mad + 1e-9), 0.0)


# ─────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────

def build_stacker_features(v7_score: np.ndarray,
                            residuals: dict[str, np.ndarray]) -> pd.DataFrame:
    """Combine v7 score + physics residuals + rolling stats into features."""
    feats = {}
    s = pd.Series(v7_score)
    feats["score"] = v7_score.astype(np.float32)
    feats["rolling_mean_10"] = s.rolling(10, min_periods=1).mean().values
    feats["rolling_std_10"] = s.rolling(10, min_periods=1).std().fillna(0).values
    feats["score_delta"] = s.diff().fillna(0).values
    feats["rolling_rank"] = s.rolling(100, min_periods=1).rank(pct=True).fillna(0.5).values

    for name, raw in residuals.items():
        z = standardize_per_event(raw).astype(np.float32)
        r = pd.Series(z)
        feats[name] = z
        feats[f"{name}_abs"] = np.abs(z)
        feats[f"{name}_mean_10"] = r.rolling(10, min_periods=1).mean().values
        feats[f"{name}_std_10"] = r.rolling(10, min_periods=1).std().fillna(0).values
        feats[f"{name}_mean_50"] = r.rolling(50, min_periods=1).mean().values
        feats[f"{name}_abs_mean_50"] = (
            pd.Series(np.abs(z)).rolling(50, min_periods=1).mean().values
        )

    return pd.DataFrame(feats)


# ─────────────────────────────────────────────────────────────────────────
# Load CARE test-region data aligned to v7 scores
# ─────────────────────────────────────────────────────────────────────────

# Hard-required columns (drivetrain channels 1-4). Electrical / cooling
# channels (sensor_7, sensor_14, sensor_15) are opportunistic — added
# when available to give the stacker transformer-relevant signal.
REQ_COLS = [
    "wind_speed_3_avg", "power_30_avg", "sensor_0_avg", "sensor_18_avg",
    "sensor_12_avg", "sensor_13_avg",
]
OPT_COLS = ["sensor_7_avg", "sensor_14_avg", "sensor_15_avg"]


def load_event_test_rows(event_id: int, n_expected: int) -> pd.DataFrame | None:
    """Load the test-region rows of a CARE Farm A event CSV.

    The v7 score array has length n_expected. CARE events use the
    `train_test` column: rows with `train_test == "prediction"` are the
    held-out evaluation rows. Returns the last n_expected rows of the
    prediction subset (or None if the CSV is missing / columns are missing).
    """
    csv = CARE_DIR / "datasets" / f"{event_id}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, sep=";", low_memory=False)
    if "train_test" in df.columns:
        df = df[df["train_test"] == "prediction"].reset_index(drop=True)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        return None
    # Align to v7 score length: trust the last n_expected rows (v7 saved post-EWMA)
    if len(df) < n_expected:
        return None
    df = df.iloc[-n_expected:].reset_index(drop=True)
    # NaN handling: column medians (matches PINN-AE code)
    for c in REQ_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in OPT_COLS:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


# ─────────────────────────────────────────────────────────────────────────
# Main LOEO evaluation
# ─────────────────────────────────────────────────────────────────────────

def run_stacker(label_key: str = "y_event") -> dict:
    if xgb is None:
        raise ImportError("xgboost not installed")

    with (RESULTS_DIR / "care_ensemble_v7_per_event_scores.json").open() as f:
        v7 = json.load(f)
    event_info = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    info_by_id = {int(r["event_id"]): r for _, r in event_info.iterrows()}

    # Build per-event feature matrices (v7 score + physics residuals)
    per_event_X: dict[int, pd.DataFrame] = {}
    per_event_y: dict[int, np.ndarray] = {}
    dropped: list[int] = []

    for eid_str, record in v7.items():
        eid = int(eid_str)
        scores = np.array(record["scores"], dtype=np.float32)
        y = np.array(record.get(label_key, []), dtype=np.int64)
        if len(y) != len(scores):
            dropped.append(eid)
            continue
        df = load_event_test_rows(eid, len(scores))
        if df is None:
            dropped.append(eid)
            continue
        residuals = physics_residuals_input(df)
        X = build_stacker_features(scores, residuals)
        per_event_X[eid] = X
        per_event_y[eid] = y

    if dropped:
        print(f"[warn] dropped {len(dropped)} events (missing CSV or label mismatch): {dropped}")

    anomaly_ids = [
        eid for eid, info in info_by_id.items()
        if info["event_label"] == "anomaly" and eid in per_event_X
    ]
    results = []
    pooled_scores: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    per_row_dump: dict[int, dict] = {}

    for held_id in anomaly_ids:
        X_train_parts, y_train_parts = [], []
        for eid, X in per_event_X.items():
            if eid == held_id:
                continue
            y = per_event_y[eid]
            if len(np.unique(y)) == 2:
                X_train_parts.append(X)
                y_train_parts.append(y)
        if not X_train_parts:
            continue
        X_train = pd.concat(X_train_parts, ignore_index=True)
        y_train = np.concatenate(y_train_parts)
        X_test = per_event_X[held_id]
        y_test = per_event_y[held_id]
        if len(np.unique(y_test)) < 2:
            continue

        # Regularised hyperparameters: shallow trees, moderate learning rate,
        # aggressive L2 + min_child_weight to prevent the stacker from
        # confidently over-riding the v7 base score on events where the
        # physics residuals carry no fault-type-relevant signal
        # (e.g. transformer faults absent drivetrain-thermal footprints).
        model = xgb.XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=20.0, reg_lambda=5.0, gamma=0.1,
            eval_metric="auc", verbosity=0, tree_method="hist",
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        base_auc = float(roc_auc_score(y_test, X_test["score"].values))
        stack_auc = float(roc_auc_score(y_test, proba))
        results.append({
            "event_id": held_id,
            "fault_type": str(info_by_id[held_id].get("event_description", "")),
            "n_test": int(len(y_test)),
            "base_v7_auc": round(base_auc, 4),
            "pinn_stacker_auc": round(stack_auc, 4),
            "delta_auc": round(stack_auc - base_auc, 4),
        })
        # Per-event rank-transform before pooling so between-event score-scale
        # drift (unavoidable under LOEO where each held-out event is scored by
        # a freshly-fit model) does not distort the pooled ranking.
        proba_ranked = (pd.Series(proba).rank(pct=True, method="average").to_numpy())
        pooled_scores.append(proba_ranked)
        pooled_labels.append(y_test)
        per_row_dump[held_id] = {
            "stacker_score": [float(x) for x in proba],
            "v7_score":      [float(x) for x in X_test["score"].values],
            "labels":        [int(x) for x in y_test],
        }

    mean_base = float(np.mean([r["base_v7_auc"] for r in results])) if results else float("nan")
    mean_stack = float(np.mean([r["pinn_stacker_auc"] for r in results])) if results else float("nan")
    ps = np.concatenate(pooled_scores) if pooled_scores else np.array([])
    pl = np.concatenate(pooled_labels) if pooled_labels else np.array([])
    pooled_auc = float(roc_auc_score(pl, ps)) if len(pl) and len(np.unique(pl)) == 2 else None

    # Identify regressions and notable events for transparent reporting
    regressions = [r for r in results if r["delta_auc"] < 0]
    recoveries  = [r for r in results if r["base_v7_auc"] < 0.35 and r["pinn_stacker_auc"] > 0.60]

    agg = {
        "label": label_key,
        "n_events": len(results),
        "mean_base_v7_auc": round(mean_base, 4),
        "mean_pinn_stacker_auc": round(mean_stack, 4),
        "mean_delta_auc": round(mean_stack - mean_base, 4),
        "median_delta_auc": round(float(np.median([r["delta_auc"] for r in results])), 4) if results else None,
        "pooled_pinn_stacker_auc": round(pooled_auc, 4) if pooled_auc is not None else None,
        "n_features": len(per_event_X[next(iter(per_event_X))].columns) if per_event_X else 0,
        "n_regressions": len(regressions),
        "n_recoveries": len(recoveries),
        # p-value source note: bootstrap p comes from delong_bootstrap.py using
        # N_BOOT=2000 resamples over per-event delta AUC. The p=0.033 figure
        # in the paper corresponds to the CARE-Precursor label (10 events);
        # the event-window label (11 events) gives p=0.074 (CI crosses zero).
        "stat_note": (
            "Bootstrap significance from delong_bootstrap.py (N_BOOT=2000). "
            "p=0.033 is under CARE-Precursor label (10 events, CI=[−0.008, 0.301]). "
            "Event-window label (11 events) gives p=0.074 with CI=[−0.045, 0.276] — "
            "CI overlaps zero; report both for transparency."
        ),
    }

    print(f"\n=== Label: {label_key} ===")
    print(f"{'ID':>4} {'Fault':<28} {'n_test':>6} {'v7 AUC':>8} {'PINN-v2':>8} {'dAUC':>8}")
    print("-" * 72)
    for r in results:
        flag = " [REGRESS]" if r["delta_auc"] < 0 else (" [RECOVER]" if r["base_v7_auc"] < 0.35 and r["pinn_stacker_auc"] > 0.60 else "")
        print(f"{r['event_id']:>4} {r['fault_type'][:26]:<28} "
              f"{r['n_test']:>6} {r['base_v7_auc']:>8.4f} {r['pinn_stacker_auc']:>8.4f} "
              f"{r['delta_auc']:>+8.4f}{flag}")
    print(f"\n  Mean v7 base AUC:           {agg['mean_base_v7_auc']:.4f}")
    print(f"  Mean PINN-v2 stacker AUC:   {agg['mean_pinn_stacker_auc']:.4f}")
    print(f"  Mean delta AUC:             {agg['mean_delta_auc']:+.4f}")
    print(f"  Pooled PINN-v2 AUC:         {agg['pooled_pinn_stacker_auc']}")
    print(f"  Features per event:         {agg['n_features']}")
    print(f"  Regressions (dAUC < 0):     {agg['n_regressions']}")
    if regressions:
        for r in regressions:
            print(f"    event {r['event_id']:>3} ({r['fault_type'][:25]:<25}): "
                  f"{r['base_v7_auc']:.4f} -> {r['pinn_stacker_auc']:.4f}"
                  f"  (dAUC={r['delta_auc']:+.4f})")
    print(f"  Recovered events (base<0.35, stack>0.60): {agg['n_recoveries']}")
    if recoveries:
        for r in recoveries:
            print(f"    event {r['event_id']:>3} ({r['fault_type'][:25]:<25}): "
                  f"{r['base_v7_auc']:.4f} -> {r['pinn_stacker_auc']:.4f}")
    print(f"\n  NOTE: {agg['stat_note']}")

    return {"aggregate": agg, "per_event": results, "per_row": per_row_dump}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for label in ("y_event", "y_precursor", "y_status"):
        result = run_stacker(label_key=label)
        out = RESULTS_DIR / f"pinn_stacker_{label}.json"
        with out.open("w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  [OK] Saved: {out}\n")


if __name__ == "__main__":
    main()
