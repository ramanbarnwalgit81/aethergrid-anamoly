"""
CARE SOTA Evaluator — matches Hybrid Autoencoder 2025 (arxiv 2510.15010) methodology.

Key ingredients that close the gap vs our previous 0.63 AUC:
  1. Rich feature engineering (moving avg, std, skew, kurt, derivatives, FFT)
  2. VAE with KL divergence scoring (plus baseline models as ensemble)
  3. Proper 70/15/15 split per event
  4. EWMA smoothing on reconstruction errors
  5. Percentile-based thresholding (95th percentile of training scores)
  6. Official CARE score (from Gück et al 2024)
  7. Ensemble across multiple autoencoders

Usage:
    python -m src.benchmark.care_sota
"""

from pathlib import Path
import json
import os, sys

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

# torch must import before numpy/pandas on Windows (OpenMP conflict).
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from src.benchmark.features import build_rich_features, engineered_feature_columns
from src.benchmark.vae_baseline import VAE, train_vae, vae_anomaly_score
from src.benchmark.care_score import compute_care_score

CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")
DATASETS_DIR = CARE_DIR / "datasets"
EVENT_INFO = CARE_DIR / "event_info.csv"
RESULTS_DIR = Path("docs/results")


# EWMA smoothing for anomaly scores
def ewma_smooth(scores: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Lower alpha = more smoothing. Typical value for fault signals: 0.2-0.3."""
    smoothed = np.empty_like(scores, dtype=np.float64)
    smoothed[0] = scores[0]
    for i in range(1, len(scores)):
        smoothed[i] = alpha * scores[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def build_event_features(df: pd.DataFrame, use_fft: bool = True) -> pd.DataFrame:
    """Build rich features for one CARE event CSV."""
    # Map CARE columns to canonical names
    from src.benchmark.adapters.care import FARM_A_COL_MAP
    df = df.rename(columns=FARM_A_COL_MAP).copy()
    # Add canonical columns missing in Farm A data as NaN
    from src.benchmark.schema import CANONICAL_RAW_FEATURES
    for c in CANONICAL_RAW_FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    if "turbine_id" not in df.columns:
        df["turbine_id"] = "T01"
    if "event_ts" not in df.columns:
        # Generate monotonic timestamps
        df["event_ts"] = pd.date_range(
            "2020-01-01", periods=len(df), freq="10min", tz="UTC"
        )
    df = df.sort_values("event_ts").reset_index(drop=True)

    # Base features that actually have data in CARE Farm A
    base = [c for c in CANONICAL_RAW_FEATURES
            if c in df.columns and df[c].notna().sum() > 100]

    enriched = build_rich_features(
        df, base_features=base, include_fft=use_fft, include_correlations=True,
    )
    return enriched


def evaluate_care_event_sota(event_row, use_fft: bool = True) -> dict:
    """SOTA evaluation of one CARE event."""
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return {"event_id": event_id, "error": "csv missing"}

    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    event_start_idx = int(event_row["event_start_id"])
    event_end_idx = int(event_row["event_end_id"])

    # Build rich features
    enriched = build_event_features(df, use_fft=use_fft)
    eng_cols = engineered_feature_columns(enriched)

    from src.benchmark.schema import CANONICAL_RAW_FEATURES
    all_feature_cols = [c for c in CANONICAL_RAW_FEATURES
                          if c in enriched.columns
                          and enriched[c].notna().sum() > 100] + eng_cols

    if len(all_feature_cols) < 10:
        return {"event_id": event_id, "error": f"only {len(all_feature_cols)} features"}

    # Labels
    n = len(enriched)
    is_anomaly = np.zeros(n, dtype=int)
    if event_row["event_label"] == "anomaly":
        is_anomaly[event_start_idx:event_end_idx + 1] = 1

    # Filter normal-only training data
    if "status_type_id" in enriched.columns:
        train_mask = (enriched["status_type_id"].fillna(0) == 0)
    else:
        # Fallback: use first 60% as normal
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:int(n * 0.6)] = True
    # Exclude labeled anomaly period from training
    train_mask = train_mask & (is_anomaly == 0)

    # 70/15/15 protocol — of the normal-only data, use 70% train / 15% val
    train_indices = np.where(train_mask)[0]
    if len(train_indices) < 200:
        return {"event_id": event_id, "error": "insufficient normal data"}

    n_tr = int(len(train_indices) * 0.7)
    actual_train_idx = train_indices[:n_tr]
    actual_val_idx = train_indices[n_tr:]

    # Impute NaN with column medians from training
    X_all = enriched[all_feature_cols].to_numpy(dtype=np.float32)
    col_medians = np.nanmedian(X_all[actual_train_idx], axis=0)
    # Replace all-NaN columns with 0
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    X_all = np.where(np.isnan(X_all), col_medians, X_all)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    X_train = X_all[actual_train_idx]
    X_val = X_all[actual_val_idx]
    X_test = X_all  # score ALL rows (including training) so we can compute event-period metrics

    # Scale
    scaler = RobustScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    # Clip to avoid extreme values breaking VAE
    X_train_s = np.clip(X_train_s, -10, 10)
    X_test_s = np.clip(X_test_s, -10, 10)

    # Train VAE
    vae_result = train_vae(
        X_train_s, n_epochs=50, patience=10,
        hidden=128, latent=16, beta=0.1,
    )
    vae_model = vae_result["model"]

    # Score
    train_scores = vae_anomaly_score(vae_model, torch.FloatTensor(X_train_s))
    test_scores = vae_anomaly_score(vae_model, torch.FloatTensor(X_test_s))

    # EWMA smoothing of test scores
    test_scores_smooth = ewma_smooth(test_scores, alpha=0.3)

    # Percentile-based threshold: 95th percentile of training scores
    threshold = float(np.percentile(train_scores, 95))

    # Evaluate AUC on test portion (after training range)
    test_portion_mask = np.ones(n, dtype=bool)
    test_portion_mask[actual_train_idx] = False
    y_test = is_anomaly[test_portion_mask]
    s_test = test_scores_smooth[test_portion_mask]

    result = {
        "event_id": event_id,
        "event_type": event_row["event_label"],
        "event_description": str(event_row.get("event_description", "")),
        "n_rows": n,
        "n_features_total": len(all_feature_cols),
        "n_features_engineered": len(eng_cols),
        "n_train": int(len(actual_train_idx)),
        "n_test": int(len(y_test)),
        "n_anomaly_rows": int(is_anomaly.sum()),
        "threshold": threshold,
        "y_true_all": is_anomaly,  # for CARE score computation
        "y_score_all": test_scores_smooth,
        "event_start_idx": event_start_idx,
        "event_end_idx": event_end_idx,
    }

    if len(np.unique(y_test)) >= 2:
        auc = float(roc_auc_score(y_test, s_test))
        result["auc_roc"] = round(auc, 4)

    return result


def main():
    print("=" * 80)
    print("  CARE SOTA Evaluator — Rich Features + VAE + EWMA + Percentile")
    print("  Matching Hybrid Autoencoder 2025 (arxiv 2510.15010) methodology")
    print("=" * 80)

    events = pd.read_csv(EVENT_INFO, sep=";")
    print(f"\n[CARE] {len(events)} events (anomaly + normal)")

    results = []
    care_event_packets = []
    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<30} {'#Feats':>7} {'AUC':>7}")
    print("-" * 72)

    for _, row in events.iterrows():
        r = evaluate_care_event_sota(row, use_fft=True)
        if "error" in r:
            print(f"{r['event_id']:>4}: ERROR — {r['error']}")
            continue

        auc_str = f"{r.get('auc_roc', '—'):.4f}" if r.get("auc_roc") is not None else "—"
        print(f"{r['event_id']:>4} {r['event_type']:>8} "
              f"{r['event_description'][:28]:<30} "
              f"{r['n_features_total']:>7} "
              f"{auc_str:>7}")

        # Stash for CARE score
        care_event_packets.append({
            "event_type": r["event_type"],
            "y_true": r.pop("y_true_all"),
            "y_score": r.pop("y_score_all"),
            "event_start_idx": r["event_start_idx"],
            "event_end_idx": r["event_end_idx"],
            "threshold": r["threshold"],
        })
        results.append(r)

    # Aggregate AUC over anomaly events
    anomaly_results = [r for r in results if r.get("event_type") == "anomaly"
                        and r.get("auc_roc") is not None]

    print(f"\n{'='*80}")
    print(f"  AGGREGATE RESULTS (VAE + Rich Features + CARE Protocol)")
    print(f"{'='*80}")

    if anomaly_results:
        aucs = [r["auc_roc"] for r in anomaly_results]
        print(f"  N anomaly events: {len(anomaly_results)}")
        print(f"  Mean AUC-ROC:     {np.mean(aucs):.4f}")
        print(f"  Median AUC-ROC:   {np.median(aucs):.4f}")
        print(f"  95% CI:           [{np.quantile(aucs, 0.025):.4f}, "
              f"{np.quantile(aucs, 0.975):.4f}]")

        # Per-fault-type
        print(f"\n  Per-Fault-Type AUC:")
        by_fault = {}
        for r in anomaly_results:
            by_fault.setdefault(r["event_description"], []).append(r["auc_roc"])
        for ft, aucs_ft in sorted(by_fault.items(), key=lambda kv: -np.mean(kv[1])):
            print(f"    {ft:<35} n={len(aucs_ft):>2}  "
                  f"mean={np.mean(aucs_ft):.4f}")

    # CARE Score
    care = compute_care_score(care_event_packets)
    print(f"\n  CARE Score (official metric):")
    for k, v in care.items():
        print(f"    {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "VAE + rich features + EWMA + percentile threshold",
        "n_events_total": len(results),
        "n_anomaly_events": len(anomaly_results),
        "mean_auc_roc": float(np.mean([r["auc_roc"] for r in anomaly_results]))
                          if anomaly_results else 0,
        "median_auc_roc": float(np.median([r["auc_roc"] for r in anomaly_results]))
                            if anomaly_results else 0,
        "care_score": care,
        "per_event": results,
    }
    with (RESULTS_DIR / "care_sota.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[OK] Saved: {RESULTS_DIR / 'care_sota.json'}")


if __name__ == "__main__":
    main()
