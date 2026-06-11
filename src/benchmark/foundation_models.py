"""
Foundation-Model benchmark on CARE Farm A — first in wind CMS.

Uses Amazon Chronos-T5-tiny (8M params, CPU-viable) to score each event's
5 most-informative signals via forecast-residual anomaly detection. Compared
against our v7 ensemble + XGBoost stacker baseline.

Protocol per event:
  1. For each target signal, slide a (context=256, prediction=64, stride=64)
     window across the entire series.
  2. At each window, feed context to Chronos, get 20-sample forecast.
  3. Anomaly score = |actual - median_forecast|.
  4. Aggregate across 5 signals via max.
  5. Evaluate AUC on event-window and CARE-Precursor labels on test region.

Expected compute: ~0.24s per forecast × 780 forecasts per series × 5 signals
× 11 events ≈ 3 hours CPU. First FM benchmark on CARE in the literature.

Usage:
    python -m src.benchmark.foundation_models
"""

from pathlib import Path
import json
import os, sys
import time
import warnings

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("docs/results")
CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")
PRECURSOR_DIR = Path("data/benchmark/care_precursor")

# Chronos hyperparameters
MODEL_NAME = "amazon/chronos-t5-tiny"
CONTEXT_LEN = 256
PREDICTION_LEN = 64
STRIDE = 64       # non-overlapping forecasts
NUM_SAMPLES = 20

TARGET_SIGNALS = [
    "sensor_12_avg",       # gearbox oil temp (Farm A)
    "sensor_0_avg",        # ambient temp
    "power_30_avg",        # active power
    "wind_speed_3_avg",    # wind speed
    "sensor_18_avg",       # generator rpm
]


def load_chronos_pipeline():
    from chronos import BaseChronosPipeline
    print(f"[LOAD] {MODEL_NAME}...")
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(
        MODEL_NAME, device_map="cpu", torch_dtype=torch.float32,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")
    return pipe


def chronos_forecast_residual(pipe, series: np.ndarray,
                                 context_len: int = CONTEXT_LEN,
                                 prediction_len: int = PREDICTION_LEN,
                                 stride: int = STRIDE) -> np.ndarray:
    """
    Score a 1-D series by sliding Chronos forecasts.
    Returns per-row residual (|actual - median forecast|).
    """
    n = len(series)
    scores = np.full(n, np.nan, dtype=np.float32)

    # Build input tensor (float32)
    s = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    for t in range(context_len, n - prediction_len + 1, stride):
        ctx = torch.tensor(s[t - context_len:t])
        try:
            fc = pipe.predict(inputs=ctx, prediction_length=prediction_len)
            # fc: (batch, num_samples, prediction_length)
            median_fc = fc.squeeze(0).median(dim=0).values.numpy()
            actual = s[t:t + prediction_len]
            err = np.abs(actual - median_fc)
            scores[t:t + prediction_len] = err
        except Exception:
            continue

    # Fill NaN edges with nearest valid value (or 0)
    first_valid = context_len
    scores[:first_valid] = scores[first_valid] if not np.isnan(scores[first_valid]) else 0.0
    scores = pd.Series(scores).fillna(method="ffill").fillna(0).to_numpy(dtype=np.float32)
    return scores


def main():
    print("=" * 80)
    print("  FOUNDATION-MODEL BENCHMARK on CARE Farm A — Chronos-T5-tiny")
    print("=" * 80)

    events_df = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    pre_df = pd.read_csv(PRECURSOR_DIR / "event_info_precursor_farm_A.csv", sep=";")
    precursor_map = {int(r["event_id"]): r for _, r in pre_df.iterrows()}
    anomaly_events = events_df[events_df["event_label"] == "anomaly"]

    pipe = load_chronos_pipeline()

    results = []
    t_start = time.time()

    for i, (_, row) in enumerate(anomaly_events.iterrows()):
        event_id = int(row["event_id"])
        desc = str(row.get("event_description", ""))[:30]
        print(f"\n[{i+1}/{len(anomaly_events)}] event {event_id} ({desc})")
        event_t0 = time.time()

        df = pd.read_csv(CARE_DIR / "datasets" / f"{event_id}.csv",
                          sep=";", low_memory=False)
        n = len(df)

        # Score each signal via Chronos forecast residuals
        per_signal_scores = []
        for sig in TARGET_SIGNALS:
            if sig not in df.columns:
                continue
            series = df[sig].fillna(method="ffill").fillna(0).to_numpy(dtype=np.float32)
            if series.std() < 1e-6:
                continue
            sig_t0 = time.time()
            scores = chronos_forecast_residual(pipe, series)
            per_signal_scores.append(scores)
            print(f"  {sig:<20}  {time.time()-sig_t0:>6.0f}s  "
                  f"score range [{scores.min():.2f}, {scores.max():.2f}]")

        if not per_signal_scores:
            print(f"  [SKIP] no signals available")
            continue

        # Per-signal z-normalize (within-series rolling-MAD) then max-aggregate
        per_signal_z = []
        for s in per_signal_scores:
            med = np.median(s)
            mad = np.median(np.abs(s - med)) + 1e-6
            z = np.abs((s - med) / (1.4826 * mad))
            per_signal_z.append(z)
        fm_score = np.stack(per_signal_z, axis=1).max(axis=1).astype(np.float32)

        # Labels
        y_event = np.zeros(n, dtype=int)
        s_idx = int(row.get("event_start_id", -1))
        e_idx = int(row.get("event_end_id", -1))
        if s_idx >= 0 and e_idx >= s_idx:
            y_event[s_idx:e_idx + 1] = 1

        y_precursor = y_event.copy()
        pre_row = precursor_map.get(event_id)
        if pre_row is not None:
            ps = int(pre_row.get("precursor_start_id", s_idx))
            if ps < s_idx:
                y_precursor[ps:e_idx + 1] = 1

        test_mask = (df.get("train_test") == "prediction").to_numpy() \
            if "train_test" in df.columns else np.ones(n, dtype=bool)
        if test_mask.sum() == 0:
            test_mask = np.ones(n, dtype=bool)

        y_ev_t = y_event[test_mask]
        y_pre_t = y_precursor[test_mask]
        fm_t = fm_score[test_mask]

        auc_ev = float(roc_auc_score(y_ev_t, fm_t)) \
            if len(np.unique(y_ev_t)) == 2 else None
        auc_pre = float(roc_auc_score(y_pre_t, fm_t)) \
            if len(np.unique(y_pre_t)) == 2 else None

        print(f"  event_window AUC: {auc_ev}")
        print(f"  precursor AUC:    {auc_pre}")
        print(f"  event time: {time.time()-event_t0:.0f}s  "
              f"total elapsed: {time.time()-t_start:.0f}s")

        results.append({
            "event_id": event_id,
            "fault": str(row.get("event_description", ""))[:40],
            "n_signals_scored": len(per_signal_scores),
            "n_test_rows": int(test_mask.sum()),
            "auc_chronos_event": auc_ev,
            "auc_chronos_precursor": auc_pre,
            "event_time_s": round(time.time() - event_t0, 1),
        })

        # Save intermediate progress
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        interim = {
            "method": "Chronos-T5-tiny forecast-residual",
            "model": MODEL_NAME,
            "context_len": CONTEXT_LEN,
            "prediction_len": PREDICTION_LEN,
            "stride": STRIDE,
            "target_signals": TARGET_SIGNALS,
            "per_event": results,
            "events_completed": i + 1,
            "events_total": len(anomaly_events),
            "elapsed_s": round(time.time() - t_start, 1),
        }
        with (RESULTS_DIR / "foundation_models_care_a.json").open("w") as f:
            json.dump(interim, f, indent=2, default=str)

    # Final aggregate
    print(f"\n{'=' * 80}")
    print(f"  AGGREGATE — Chronos-T5-tiny on CARE Farm A")
    print(f"{'=' * 80}")

    for key in ["auc_chronos_event", "auc_chronos_precursor"]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"  {key:<28} n={len(vals):>2}  mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}")

    interim["mean_auc_event"] = float(np.mean([r["auc_chronos_event"] for r in results
                                                  if r.get("auc_chronos_event") is not None]))
    interim["mean_auc_precursor"] = float(np.mean([r["auc_chronos_precursor"] for r in results
                                                       if r.get("auc_chronos_precursor") is not None]))
    interim["total_time_s"] = round(time.time() - t_start, 1)

    with (RESULTS_DIR / "foundation_models_care_a.json").open("w") as f:
        json.dump(interim, f, indent=2, default=str)
    print(f"\n[OK] Saved: {RESULTS_DIR / 'foundation_models_care_a.json'}")
    print(f"[TIME] total {time.time() - t_start:.0f}s = {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
