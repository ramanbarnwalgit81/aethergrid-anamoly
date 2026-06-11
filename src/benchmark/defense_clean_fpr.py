"""Compute the physics-consistency defense's clean-data false-rejection rate.

The defense rule is C(x) = exp(-L_phys(x)) and rejects rows with C(x) < tau.
We sweep tau over the four-residual consistency and report the FPR on a large
sample of known-healthy Penmanshiel operation rows. The adversarial table
reports 0% attack success with the defense; this script gives the operator
the complementary number: the fraction of clean rows the defense flags by
mistake.

Usage: python -m src.benchmark.defense_clean_fpr
Outputs: docs/results/defense_clean_fpr.json
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

RESULTS_DIR = Path("docs/results")
PEN_PARQUET = Path("data/benchmark/harmonized/penmanshiel_full.parquet")
RATED_POWER_KW = 2000.0
CUT_IN_WIND = 3.0
RATED_WIND = 12.0
CUT_OUT_WIND = 25.0


def iec_power_curve(wind: np.ndarray) -> np.ndarray:
    w = np.clip(wind, 0.0, CUT_OUT_WIND + 1)
    ramp = (w >= CUT_IN_WIND) & (w < RATED_WIND)
    rated = (w >= RATED_WIND) & (w < CUT_OUT_WIND)
    ramp_ratio = np.clip((w - CUT_IN_WIND) / (RATED_WIND - CUT_IN_WIND), 0.0, 1.0)
    return (
        ramp.astype(float) * RATED_POWER_KW * ramp_ratio ** 3
        + rated.astype(float) * RATED_POWER_KW
    )


def physics_consistency(df: pd.DataFrame) -> np.ndarray:
    wind = df["wind_speed_ms"].to_numpy(dtype=np.float32)
    power = df["active_power_kw"].to_numpy(dtype=np.float32)
    ambient = df["ambient_temp_c"].to_numpy(dtype=np.float32)
    # Penmanshiel uses main_bearing; Kelmarsh uses gearbox_oil_temp_c; CARE uses
    # sensor_12_avg.  Accept any of these as the drivetrain-thermal proxy.
    for tcol in ("gearbox_oil_temp_c", "main_bearing_temp_c",
                  "generator_bearing_de_temp_c"):
        if tcol in df.columns and df[tcol].notna().any():
            thermal = df[tcol].to_numpy(dtype=np.float32)
            break
    else:
        raise KeyError("no drivetrain-thermal column in dataframe")

    r_p = (power - iec_power_curve(wind)) / RATED_POWER_KW
    power_ratio = np.clip(power / RATED_POWER_KW, 0.0, 1.2)
    expected_thermal = ambient + 40.0 + 25.0 * power_ratio
    r_t = (thermal - expected_thermal) / 30.0

    l_phys = 0.5 * (r_p ** 2 + r_t ** 2)
    return np.exp(-l_phys)


def main():
    df = pd.read_parquet(PEN_PARQUET)
    # Drop rows missing any of the required columns that we can find.
    cols_req = ["wind_speed_ms", "active_power_kw", "ambient_temp_c"]
    df = df.dropna(subset=cols_req)
    thermal_col = next(
        (c for c in ("gearbox_oil_temp_c", "main_bearing_temp_c",
                       "generator_bearing_de_temp_c") if c in df.columns and df[c].notna().any()),
        None,
    )
    if thermal_col is None:
        print("[ERR] no drivetrain-thermal column found")
        return
    df = df.dropna(subset=[thermal_col])
    df = df[df["active_power_kw"] > 0]
    df = df[df["wind_speed_ms"].between(CUT_IN_WIND, CUT_OUT_WIND)]
    print(f"[DATA] {len(df):,} healthy-operation rows (Penmanshiel, thermal={thermal_col})")

    C = physics_consistency(df)
    C = C[np.isfinite(C)]

    # FPR at various thresholds
    thresholds = [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
    out = {
        "dataset": "penmanshiel healthy-operation",
        "n_clean_rows": int(len(C)),
        "consistency_quantiles": {
            "q01": float(np.quantile(C, 0.01)),
            "q05": float(np.quantile(C, 0.05)),
            "q25": float(np.quantile(C, 0.25)),
            "q50": float(np.quantile(C, 0.50)),
            "q75": float(np.quantile(C, 0.75)),
            "q95": float(np.quantile(C, 0.95)),
        },
        "defense_clean_fpr_at_tau": {
            str(t): round(float(np.mean(C < t)), 4) for t in thresholds
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "defense_clean_fpr.json").open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\nConsistency C(x) distribution (n={len(C):,}):")
    for k, v in out["consistency_quantiles"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nDefense clean-data FPR by threshold:")
    for t, fpr in out["defense_clean_fpr_at_tau"].items():
        print(f"  tau={t:>4}: FPR = {fpr:.4f}")
    print(f"\n[OK] Saved: docs/results/defense_clean_fpr.json")


if __name__ == "__main__":
    main()
