"""Cross-farm PINN-v2 pooling on all 44 CARE anomaly events.

The paper's Farm-A-only PINN-v2 evaluation achieves a 12-point mean AUC
uplift over v7 base with a wide bootstrap CI ($[-0.05,+0.28]$, p≈0.07,
n=11). This script extends the stacker to Farms B and C so the same
paired-delta bootstrap runs on n=44, pushing the claim from "directional"
to "statistically significant at alpha=0.05" if the uplift transfers.

FARM-SPECIFIC PHYSICS CHANNELS
------------------------------
Only the IEC power-curve residual is reliably mappable across all three
farms (each has at least one labelled `power_*_avg` and `wind_speed_*_avg`
column). Thermal channels are Farm A-specific because the CARE anonymised
sensor layout does not disclose which `sensor_N` is gearbox-oil on B or C.

We therefore run a "conservative PINN-v2" that uses ONE physics channel
(power-curve) plus the v7 base score and their rolling temporal stats --
11 features total, shared across all three farms. Full 7-channel PINN-v2
results on Farm A are reported alongside as a comparison; the cross-farm
pooled number is the conservative one.

Output:  docs/results/pinn_stacker_crossfarm.json
Usage:   python -m src.benchmark.pinn_stacker_crossfarm
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
CARE_ROOT = Path("data/real_scada/care/extracted")
RATED_POWER_KW = 2000.0  # only used for Farm A normalisation
CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0
RNG = np.random.default_rng(42)
N_BOOT = 2000

# Farm-specific physics column mapping. All columns are `*_avg` suffixed.
FARM_CFG = {
    "A": {
        "csv_dir":  CARE_ROOT / "Wind Farm A/Wind Farm A/datasets",
        "event_info": CARE_ROOT / "Wind Farm A/Wind Farm A/event_info.csv",
        "wind":  "wind_speed_3_avg",
        "power": "power_30_avg",
        "power_unit": "kw",       # divide by RATED_POWER_KW to get fraction
    },
    "B": {
        "csv_dir":  CARE_ROOT / "Wind Farm B/Wind Farm B/datasets",
        "event_info": CARE_ROOT / "Wind Farm B/Wind Farm B/event_info.csv",
        "wind":  "wind_speed_59_avg",
        "power": "power_58_avg",
        "power_unit": "fraction",  # already normalised to [0,1]
    },
    "C": {
        "csv_dir":  CARE_ROOT / "Wind Farm C/Wind Farm C/datasets",
        "event_info": CARE_ROOT / "Wind Farm C/Wind Farm C/event_info.csv",
        "wind":  "wind_speed_236_avg",
        "power": "power_2_avg",
        "power_unit": "fraction",
    },
}


def iec_power_curve_fraction(wind: np.ndarray) -> np.ndarray:
    """IEC cubic ramp, returning fraction of rated (0..1)."""
    w = np.clip(wind, 0.0, CUT_OUT + 1)
    ramp = (w >= CUT_IN) & (w < RATED)
    rated = (w >= RATED) & (w < CUT_OUT)
    r = np.clip((w - CUT_IN) / (RATED - CUT_IN), 0.0, 1.0)
    return ramp.astype(float) * r ** 3 + rated.astype(float) * 1.0


def power_curve_residual(df: pd.DataFrame, farm: str) -> np.ndarray:
    cfg = FARM_CFG[farm]
    wind = df[cfg["wind"]].to_numpy(dtype=np.float32)
    power_raw = df[cfg["power"]].to_numpy(dtype=np.float32)
    p_frac = power_raw if cfg["power_unit"] == "fraction" else power_raw / RATED_POWER_KW
    return (p_frac - iec_power_curve_fraction(wind)).astype(np.float32)


def robust_z(x: np.ndarray) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return np.zeros_like(x)
    med = np.median(finite)
    mad = 1.4826 * np.median(np.abs(finite - med))
    if mad < 1e-9:
        mad = np.std(finite) if len(finite) > 1 else 1.0
    return np.where(np.isfinite(x), (x - med) / (mad + 1e-9), 0.0).astype(np.float32)


def build_features(v7_score: np.ndarray, r_power: np.ndarray) -> pd.DataFrame:
    """11-feature shared-across-farms vector."""
    s = pd.Series(v7_score)
    z = robust_z(r_power)
    zs = pd.Series(z)
    return pd.DataFrame({
        "score":            v7_score.astype(np.float32),
        "score_roll_m10":   s.rolling(10, min_periods=1).mean().values,
        "score_roll_s10":   s.rolling(10, min_periods=1).std().fillna(0).values,
        "score_delta":      s.diff().fillna(0).values,
        "score_rank":       s.rolling(100, min_periods=1).rank(pct=True).fillna(0.5).values,
        "r_power":          z,
        "r_power_abs":      np.abs(z),
        "r_power_m10":      zs.rolling(10, min_periods=1).mean().values,
        "r_power_s10":      zs.rolling(10, min_periods=1).std().fillna(0).values,
        "r_power_m50":      zs.rolling(50, min_periods=1).mean().values,
        "r_power_abs_m50":  pd.Series(np.abs(z)).rolling(50, min_periods=1).mean().values,
    })


def load_v7_scores_farm(farm: str) -> dict:
    """Load the v7 per-event/per-row score arrays for a farm.

    All farms use the per-row score dump written by ensemble_care_farm.py, which
    shares the Farm-A schema (dict: event_id -> {scores, y_event, y_precursor,
    y_status, sf_mask}). Farm A keeps the historical filename.
    """
    if farm == "A":
        fn = RESULTS_DIR / "care_ensemble_v7_per_event_scores.json"
    else:
        fn = RESULTS_DIR / f"care_ensemble_v7_per_event_scores_{farm.lower()}.json"
    if not fn.exists():
        print(f"[WARN] {fn} missing for farm {farm}; "
              f"run: python -m src.benchmark.ensemble_care_farm --farm {farm}")
        return {}
    with fn.open() as f:
        return json.load(f)


def load_event_test_rows(farm: str, event_id: int, n_expected: int) -> pd.DataFrame | None:
    """Load test-region rows aligned to the v7 score array."""
    cfg = FARM_CFG[farm]
    csv = cfg["csv_dir"] / f"{event_id}.csv"
    if not csv.exists():
        return None
    _need = {cfg["wind"], cfg["power"], "train_test"}
    df = pd.read_csv(csv, sep=";", low_memory=False,
                     usecols=lambda c: c in _need)
    if "train_test" in df.columns:
        df = df[df["train_test"] == "prediction"].reset_index(drop=True)
    for col in (cfg["wind"], cfg["power"]):
        if col not in df.columns:
            return None
    if len(df) < n_expected:
        return None
    df = df.iloc[-n_expected:].reset_index(drop=True)
    for col in (cfg["wind"], cfg["power"]):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def run_loeo_for_farm(farm: str, label_key: str = "y_event") -> list:
    """Leave-one-event-out stacker on one farm. Returns per-event dicts."""
    v7 = load_v7_scores_farm(farm)
    event_info = pd.read_csv(FARM_CFG[farm]["event_info"], sep=";")
    info_by_id = {int(r["event_id"]): r for _, r in event_info.iterrows()}

    per_event_X, per_event_y = {}, {}
    for eid_str, record in v7.items():
        eid = int(eid_str)
        scores = np.asarray(record["scores"], dtype=np.float32)
        if label_key not in record:
            continue
        y = np.asarray(record[label_key], dtype=np.int64)
        if len(y) != len(scores) or len(np.unique(y)) < 2:
            continue
        df_test = load_event_test_rows(farm, eid, len(scores))
        if df_test is None:
            continue
        r = power_curve_residual(df_test, farm)
        X = build_features(scores, r)
        per_event_X[eid] = X
        per_event_y[eid] = y

    anomaly_ids = [eid for eid, info in info_by_id.items()
                    if info["event_label"] == "anomaly" and eid in per_event_X]

    per_event_results = []
    for held in anomaly_ids:
        X_train_parts, y_train_parts = [], []
        for eid, X in per_event_X.items():
            if eid == held:
                continue
            y = per_event_y[eid]
            if len(np.unique(y)) == 2:
                X_train_parts.append(X)
                y_train_parts.append(y)
        if not X_train_parts:
            continue
        X_train = pd.concat(X_train_parts, ignore_index=True)
        y_train = np.concatenate(y_train_parts)
        X_test = per_event_X[held]
        y_test = per_event_y[held]

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
        per_event_results.append({
            "farm": farm,
            "event_id": held,
            "fault_type": str(info_by_id[held].get("event_description", "")),
            "n_test": int(len(y_test)),
            "base_v7_auc": round(base_auc, 4),
            "pinn_stacker_auc": round(stack_auc, 4),
            "delta_auc": round(stack_auc - base_auc, 4),
        })
    return per_event_results


def paired_bootstrap_delta(deltas: np.ndarray, n_boot: int = N_BOOT) -> dict:
    n = len(deltas)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = deltas[RNG.integers(0, n, n)].mean()
    lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {
        "n": int(n),
        "observed_mean_delta": round(float(deltas.mean()), 4),
        "bootstrap_95ci_mean_delta": [round(lo, 4), round(hi, 4)],
        "zero_in_ci": bool(lo <= 0 <= hi),
        "p_one_sided": round(float((means <= 0).mean()), 4),
    }


def main():
    if xgb is None:
        raise ImportError("xgboost not installed")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_per_event = {"y_event": [], "y_precursor": [], "y_status": []}
    for label in all_per_event:
        print(f"\n=== Label: {label} ===")
        for farm in ("A", "B", "C"):
            print(f"[Farm {farm}] running LOEO…")
            r = run_loeo_for_farm(farm, label)
            all_per_event[label].extend(r)
            print(f"  -> {len(r)} anomaly events scored")

    summary = {}
    for label, per in all_per_event.items():
        if not per:
            summary[label] = {"error": "no events"}
            continue
        deltas = np.array([r["delta_auc"] for r in per], dtype=float)
        # also break down per farm
        per_farm = {}
        for farm in ("A", "B", "C"):
            farm_d = np.array([r["delta_auc"] for r in per if r["farm"] == farm])
            if len(farm_d) > 0:
                per_farm[farm] = paired_bootstrap_delta(farm_d)

        pooled = paired_bootstrap_delta(deltas)
        summary[label] = {
            "per_farm": per_farm,
            "pooled_across_farms": pooled,
            "n_total_events": len(per),
        }

    out = {
        "method": "Cross-farm PINN-v2 conservative (power-curve residual only, "
                    "shared-across-farms 11-feature vector)",
        "per_event": all_per_event,
        "summary": summary,
    }
    (RESULTS_DIR / "pinn_stacker_crossfarm.json").write_text(
        json.dumps(out, indent=2, default=str))

    # Print summary
    print("\n\n" + "=" * 72)
    print("CROSS-FARM PINN-v2 POOLED RESULTS")
    print("=" * 72)
    for label, s in summary.items():
        if "error" in s:
            print(f"{label}: no data"); continue
        p = s["pooled_across_farms"]
        print(f"\n{label}: pooled n={p['n']}, mean delta={p['observed_mean_delta']:+.4f}, "
              f"95% CI={p['bootstrap_95ci_mean_delta']}, one-sided p={p['p_one_sided']}")
        print("  per-farm:")
        for farm, pf in s["per_farm"].items():
            print(f"    {farm}: n={pf['n']}, delta={pf['observed_mean_delta']:+.4f}, "
                  f"CI={pf['bootstrap_95ci_mean_delta']}, p={pf['p_one_sided']}")
    print(f"\n[OK] Saved: docs/results/pinn_stacker_crossfarm.json")


if __name__ == "__main__":
    main()
