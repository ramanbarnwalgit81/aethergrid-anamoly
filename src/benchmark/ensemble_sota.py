"""
Ensemble SOTA Evaluator — VAE + LSTM-AE + Transformer-AE on CARE Farm A.

Replicates the Hybrid Autoencoder 2025 methodology (arxiv 2510.15010) that
achieved AUC 0.947 on the CARE benchmark.

Pipeline:
  1. Rich features (rolling + derivatives + FFT + correlations)
  2. RobustScaler normalization + clipping
  3. Train three complementary autoencoders on normal-only windows:
     - VAE (static anomaly via reconstruction + KL)
     - LSTM-AE (temporal dependencies)
     - Transformer-AE (long-range attention)
  4. Combine z-score normalized anomaly scores (average)
  5. EWMA smoothing (alpha=0.2 — less aggressive than before)
  6. Percentile threshold from training scores for CARE metrics
  7. AUC on test portion, CARE score on full events

Usage:
    python -m src.benchmark.ensemble_sota
"""

from pathlib import Path
import json
import os, sys

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

# torch before numpy/pandas (Windows OpenMP conflict)
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from src.benchmark.features import build_rich_features, engineered_feature_columns
from src.benchmark.vae_baseline import train_vae, vae_anomaly_score
from src.benchmark.lstm_ae import train_lstm_ae, lstm_ae_anomaly_score
from src.benchmark.transformer_ae import train_transformer_ae, transformer_ae_anomaly_score
from src.benchmark.care_score import compute_care_score
from src.benchmark.care_sota import build_event_features, ewma_smooth

CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")
DATASETS_DIR = CARE_DIR / "datasets"
EVENT_INFO = CARE_DIR / "event_info.csv"
RESULTS_DIR = Path("docs/results")

# Ensemble hyperparameters
SEQ_LEN = 24
VAE_HIDDEN = 128
VAE_LATENT = 16
LSTM_HIDDEN = 64
TRANSFORMER_DMODEL = 64
TRANSFORMER_HEADS = 4
EWMA_ALPHA = 0.2


def z_normalize(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Z-score normalize `scores` using the reference's mean and std."""
    mu = float(reference.mean())
    sigma = float(reference.std())
    if sigma < 1e-9:
        return scores - mu
    return (scores - mu) / sigma


def evaluate_event_ensemble(event_row, use_fft: bool = True) -> dict:
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return {"event_id": event_id, "error": "csv missing"}

    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    event_start_idx = int(event_row["event_start_id"])
    event_end_idx = int(event_row["event_end_id"])

    enriched = build_event_features(df, use_fft=use_fft)
    eng_cols = engineered_feature_columns(enriched)

    from src.benchmark.schema import CANONICAL_RAW_FEATURES
    all_feature_cols = [c for c in CANONICAL_RAW_FEATURES
                          if c in enriched.columns
                          and enriched[c].notna().sum() > 100] + eng_cols

    if len(all_feature_cols) < 10:
        return {"event_id": event_id, "error": f"only {len(all_feature_cols)} features"}

    n = len(enriched)
    is_anomaly = np.zeros(n, dtype=int)
    if event_row["event_label"] == "anomaly":
        is_anomaly[event_start_idx:event_end_idx + 1] = 1

    # Training mask: prefer status_type_id == 0 if available; else first 60%
    if "status_type_id" in enriched.columns:
        train_mask = (enriched["status_type_id"].fillna(0) == 0)
    else:
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:int(n * 0.6)] = True
    train_mask = train_mask & (is_anomaly == 0)

    train_indices = np.where(train_mask)[0]
    if len(train_indices) < 200:
        return {"event_id": event_id, "error": "insufficient normal data"}

    n_tr = int(len(train_indices) * 0.7)
    actual_train_idx = train_indices[:n_tr]

    X_all = enriched[all_feature_cols].to_numpy(dtype=np.float32)
    col_medians = np.nanmedian(X_all[actual_train_idx], axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    X_all = np.where(np.isnan(X_all), col_medians, X_all)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    X_train = X_all[actual_train_idx]

    scaler = RobustScaler().fit(X_train)
    X_train_s = np.clip(scaler.transform(X_train), -10, 10).astype(np.float32)
    X_all_s = np.clip(scaler.transform(X_all), -10, 10).astype(np.float32)

    # ---- VAE ----
    vae_bundle = train_vae(X_train_s, n_epochs=40, patience=8,
                             hidden=VAE_HIDDEN, latent=VAE_LATENT, beta=0.1)
    vae_model = vae_bundle["model"]
    vae_train = vae_anomaly_score(vae_model, torch.FloatTensor(X_train_s))
    vae_test = vae_anomaly_score(vae_model, torch.FloatTensor(X_all_s))

    # ---- LSTM-AE ----
    lstm_bundle = train_lstm_ae(X_train_s, seq_len=SEQ_LEN,
                                  n_epochs=20, patience=5,
                                  hidden=LSTM_HIDDEN)
    lstm_train = lstm_ae_anomaly_score(lstm_bundle, X_train_s)
    lstm_test = lstm_ae_anomaly_score(lstm_bundle, X_all_s)

    # ---- Transformer-AE ----
    tf_bundle = train_transformer_ae(X_train_s, seq_len=SEQ_LEN,
                                       n_epochs=20, patience=5,
                                       d_model=TRANSFORMER_DMODEL,
                                       n_heads=TRANSFORMER_HEADS)
    tf_train = transformer_ae_anomaly_score(tf_bundle, X_train_s)
    tf_test = transformer_ae_anomaly_score(tf_bundle, X_all_s)

    # ---- Z-normalize each component against its own train distribution ----
    vae_test_z = z_normalize(vae_test, vae_train)
    lstm_test_z = z_normalize(lstm_test, lstm_train)
    tf_test_z = z_normalize(tf_test, tf_train)

    # Ensemble: mean of z-scores
    ensemble_test = (vae_test_z + lstm_test_z + tf_test_z) / 3.0
    ensemble_train = (z_normalize(vae_train, vae_train) +
                       z_normalize(lstm_train, lstm_train) +
                       z_normalize(tf_train, tf_train)) / 3.0

    # Smoothing (less aggressive)
    ensemble_smooth = ewma_smooth(ensemble_test, alpha=EWMA_ALPHA)

    # Threshold from training ensemble
    threshold = float(np.percentile(ensemble_train, 95))

    # Evaluate AUC on test portion (rows not used in training)
    test_portion_mask = np.ones(n, dtype=bool)
    test_portion_mask[actual_train_idx] = False
    y_test = is_anomaly[test_portion_mask]
    s_test = ensemble_smooth[test_portion_mask]

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
        "y_true_all": is_anomaly,
        "y_score_all": ensemble_smooth,
        "event_start_idx": event_start_idx,
        "event_end_idx": event_end_idx,
    }

    # Per-component AUC (diagnostics)
    if len(np.unique(y_test)) >= 2:
        result["auc_ensemble"] = round(float(roc_auc_score(y_test, s_test)), 4)
        result["auc_vae"] = round(float(roc_auc_score(y_test, vae_test[test_portion_mask])), 4)
        result["auc_lstm"] = round(float(roc_auc_score(y_test, lstm_test[test_portion_mask])), 4)
        result["auc_transformer"] = round(float(roc_auc_score(y_test, tf_test[test_portion_mask])), 4)

    return result


def main():
    print("=" * 80)
    print("  ENSEMBLE SOTA — VAE + LSTM-AE + Transformer-AE on CARE Farm A")
    print("  Replicating Hybrid Autoencoder 2025 (arxiv 2510.15010)")
    print("=" * 80)

    events = pd.read_csv(EVENT_INFO, sep=";")
    print(f"\n[ENS] {len(events)} events (anomaly + normal)")

    results = []
    care_packets = []
    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<28} "
          f"{'VAE':>6} {'LSTM':>6} {'TF':>6} {'ENS':>6}")
    print("-" * 78)

    for _, row in events.iterrows():
        r = evaluate_event_ensemble(row, use_fft=True)
        if "error" in r:
            print(f"{r['event_id']:>4}: ERROR — {r['error']}")
            continue

        print(f"{r['event_id']:>4} {r['event_type']:>8} "
              f"{r['event_description'][:26]:<28} "
              f"{r.get('auc_vae', 0):>6.3f} "
              f"{r.get('auc_lstm', 0):>6.3f} "
              f"{r.get('auc_transformer', 0):>6.3f} "
              f"{r.get('auc_ensemble', 0):>6.3f}")

        care_packets.append({
            "event_type": r["event_type"],
            "y_true": r.pop("y_true_all"),
            "y_score": r.pop("y_score_all"),
            "event_start_idx": r["event_start_idx"],
            "event_end_idx": r["event_end_idx"],
            "threshold": r["threshold"],
        })
        results.append(r)

    anomaly_results = [r for r in results if r.get("event_type") == "anomaly"
                        and r.get("auc_ensemble") is not None]

    print(f"\n{'='*80}")
    print(f"  AGGREGATE RESULTS (ENSEMBLE + FFT + z-norm)")
    print(f"{'='*80}")

    if anomaly_results:
        for key in ["auc_vae", "auc_lstm", "auc_transformer", "auc_ensemble"]:
            vals = [r[key] for r in anomaly_results if r.get(key) is not None]
            print(f"  {key:<18} mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")

        # Per-fault-type
        print(f"\n  Per-Fault-Type Ensemble AUC:")
        by_fault = {}
        for r in anomaly_results:
            by_fault.setdefault(r["event_description"], []).append(r["auc_ensemble"])
        for ft, aucs_ft in sorted(by_fault.items(), key=lambda kv: -np.mean(kv[1])):
            print(f"    {ft:<35} n={len(aucs_ft):>2}  "
                  f"mean={np.mean(aucs_ft):.4f}")

    care = compute_care_score(care_packets)
    print(f"\n  CARE Score (official):")
    for k, v in care.items():
        print(f"    {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Ensemble (VAE + LSTM-AE + Transformer-AE) + FFT features + z-norm",
        "seq_len": SEQ_LEN,
        "ewma_alpha": EWMA_ALPHA,
        "n_events_total": len(results),
        "n_anomaly_events": len(anomaly_results),
        "mean_auc_ensemble": float(np.mean([r["auc_ensemble"] for r in anomaly_results]))
                               if anomaly_results else 0,
        "mean_auc_vae": float(np.mean([r["auc_vae"] for r in anomaly_results]))
                          if anomaly_results else 0,
        "mean_auc_lstm": float(np.mean([r["auc_lstm"] for r in anomaly_results]))
                           if anomaly_results else 0,
        "mean_auc_transformer": float(np.mean([r["auc_transformer"] for r in anomaly_results]))
                                  if anomaly_results else 0,
        "care_score": care,
        "per_event": results,
    }
    with (RESULTS_DIR / "care_ensemble_sota.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[OK] Saved: {RESULTS_DIR / 'care_ensemble_sota.json'}")


if __name__ == "__main__":
    main()
