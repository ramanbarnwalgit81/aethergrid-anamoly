"""WindGPT pre-training corpus builder.

Produces a harmonised healthy-operation corpus pooling Penmanshiel and
Kelmarsh SCADA under a canonical 5-signal schema. Writes:

    data/benchmark/windgpt/pretrain_corpus.parquet

Canonical signals (match the CARE-Farm-A physics roles used in PINN-v2):
    wind_speed_ms, ambient_temp_c, active_power_kw, rotor_speed_rpm,
    gearbox_oil_temp_c  (with a fallback to main_bearing_temp_c on
                         Penmanshiel, which has no gearbox_oil column)

We additionally drop rows where the turbine is NOT in healthy operation
(power > 0, cut-in <= wind < cut-out, rotor RPM > 0.5). Remaining rows are
used as forecasting windows by lora_finetune.make_training_windows().

Usage:
    python -m src.benchmark.windgpt_pretrain

The output parquet is consumed by
    python -m src.benchmark.lora_finetune train --corpus windgpt
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "benchmark" / "windgpt"
OUT_FILE = OUT_DIR / "pretrain_corpus.parquet"

PENMANSHIEL = REPO_ROOT / "data" / "benchmark" / "harmonized" / "penmanshiel_full.parquet"
KELMARSH   = REPO_ROOT / "data" / "benchmark" / "harmonized" / "kelmarsh.parquet"

CANONICAL = ["wind_speed_ms", "ambient_temp_c", "active_power_kw",
              "rotor_speed_rpm", "gearbox_oil_temp_c"]


def harmonise_penmanshiel() -> pd.DataFrame:
    if not PENMANSHIEL.exists():
        print(f"[skip] {PENMANSHIEL} not found")
        return pd.DataFrame()
    df = pd.read_parquet(PENMANSHIEL)
    # Gearbox oil is all-NaN on Penmanshiel; use main_bearing_temp_c as the
    # drivetrain-thermal proxy (same decision we make in defense_clean_fpr.py).
    if df.get("gearbox_oil_temp_c") is None or df["gearbox_oil_temp_c"].isna().all():
        if "main_bearing_temp_c" in df.columns:
            df["gearbox_oil_temp_c"] = df["main_bearing_temp_c"]
    keep = [c for c in CANONICAL if c in df.columns]
    missing = [c for c in CANONICAL if c not in df.columns]
    if missing:
        print(f"[warn] Penmanshiel missing {missing}; filling with NaN")
        for c in missing:
            df[c] = np.nan
    df = df[CANONICAL + ["turbine_id"]].copy() if "turbine_id" in df.columns \
         else df[CANONICAL].copy()
    df["farm"] = "penmanshiel"
    return df


def harmonise_kelmarsh() -> pd.DataFrame:
    if not KELMARSH.exists():
        print(f"[skip] {KELMARSH} not found")
        return pd.DataFrame()
    df = pd.read_parquet(KELMARSH)
    keep = [c for c in CANONICAL if c in df.columns]
    missing = [c for c in CANONICAL if c not in df.columns]
    if missing:
        print(f"[warn] Kelmarsh missing {missing}; filling with NaN")
        for c in missing:
            df[c] = np.nan
    df = df[CANONICAL + ["turbine_id"]].copy() if "turbine_id" in df.columns \
         else df[CANONICAL].copy()
    df["farm"] = "kelmarsh"
    return df


def filter_healthy(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    mask = (
        (df["active_power_kw"].fillna(-1) > 0)
        & (df["wind_speed_ms"].fillna(0).between(3.0, 25.0))
        & (df["rotor_speed_rpm"].fillna(0) > 0.5)
    )
    out = df[mask].copy()
    print(f"  [filter] kept {len(out):,}/{n0:,} rows ({100*len(out)/max(1,n0):.1f}%) "
          f"as healthy operation")
    return out


def main():
    print("[1/3] Harmonising Penmanshiel…")
    p = harmonise_penmanshiel()
    print(f"  -> {len(p):,} rows")
    print("[2/3] Harmonising Kelmarsh…")
    k = harmonise_kelmarsh()
    print(f"  -> {len(k):,} rows")

    all_df = pd.concat([p, k], ignore_index=True)
    all_df = filter_healthy(all_df)

    print("[3/3] Saving pooled corpus…")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(OUT_FILE, compression="zstd")
    print(f"  wrote {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")
    print(f"\nPer-signal non-NaN rows:")
    for c in CANONICAL:
        n = all_df[c].notna().sum()
        print(f"  {c:<25s} {n:>12,}")
    print(f"\nPer-farm row counts:")
    print(all_df["farm"].value_counts().to_string())


if __name__ == "__main__":
    main()
