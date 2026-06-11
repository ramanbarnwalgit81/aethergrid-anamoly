"""
Ensemble SOTA Evaluator v2 — PROPER CARE protocol.

FIX vs v1: train on pooled NORMAL events (cross-event protocol), not on the
first 70% of each event's own timeline. The previous protocol conflated
natural drift with anomaly — both train and test regions drifted together,
so the model learned "drift is normal" and flunked AUC.

New protocol (matches how Hybrid Autoencoder 2025 likely evaluates):
  1. Collect all normal events (CARE labels them event_label == "normal")
  2. Pool them into a big training matrix
  3. Train VAE + LSTM-AE + Transformer-AE on the pooled training set
  4. Score each anomaly event's full timeline
  5. AUC is computed on the entire event timeline (all rows)
  6. Per-fault-type and overall AUC reported

Usage:
    python -m src.benchmark.ensemble_sota_v2
"""

from pathlib import Path
import json
import os, sys
import time

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

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

# Ensemble hyperparameters — chosen for speed on CPU.
# SEQ_LEN=12 cuts Transformer cost 4x vs SEQ_LEN=24 with small AUC loss.
SEQ_LEN = 12
VAE_HIDDEN = 128
VAE_LATENT = 16
LSTM_HIDDEN = 64
TRANSFORMER_DMODEL = 64
TRANSFORMER_HEADS = 4
EWMA_ALPHA = 0.2


def z_normalize(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mu = float(reference.mean())
    sigma = float(reference.std())
    if sigma < 1e-9:
        return scores - mu
    return (scores - mu) / sigma


def load_event_features(event_row, use_fft: bool = True):
    """Load one event CSV, compute rich features, return (enriched, feature_cols, labels)."""
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    enriched = build_event_features(df, use_fft=use_fft)
    eng_cols = engineered_feature_columns(enriched)

    from src.benchmark.schema import CANONICAL_RAW_FEATURES
    raw_avail = [c for c in CANONICAL_RAW_FEATURES
                  if c in enriched.columns and enriched[c].notna().sum() > 100]
    feature_cols = raw_avail + eng_cols

    n = len(enriched)
    is_anomaly = np.zeros(n, dtype=int)
    if event_row["event_label"] == "anomaly":
        s = int(event_row["event_start_id"])
        e = int(event_row["event_end_id"])
        is_anomaly[s:e + 1] = 1

    return {
        "event_id": event_id,
        "event_label": event_row["event_label"],
        "event_description": str(event_row.get("event_description", "")),
        "event_start_idx": int(event_row.get("event_start_id", -1)),
        "event_end_idx": int(event_row.get("event_end_id", -1)),
        "enriched": enriched,
        "feature_cols": feature_cols,
        "is_anomaly": is_anomaly,
        "n_rows": n,
    }


def build_pooled_training_set(normal_events: list):
    """
    Collect feature matrices from all normal events, impute NaN, clip, scale.
    Returns (X_train, col_medians, scaler, feature_cols).
    """
    # Use intersection of feature columns across all normal events
    col_sets = [set(ev["feature_cols"]) for ev in normal_events if ev is not None]
    shared_cols = list(sorted(set.intersection(*col_sets))) if col_sets else []
    if not shared_cols:
        raise RuntimeError("No shared features across normal events")

    print(f"  [POOL] Shared features across normal events: {len(shared_cols)}")

    # Subsample each normal event to cap total training size (speed)
    PER_EVENT_CAP = 8000
    X_parts = []
    for ev in normal_events:
        if ev is None:
            continue
        X = ev["enriched"][shared_cols].to_numpy(dtype=np.float32)
        if len(X) > PER_EVENT_CAP:
            # Take uniform sample (preserves drift profile, avoids overweighting)
            idx = np.linspace(0, len(X) - 1, PER_EVENT_CAP).astype(int)
            X = X[idx]
        X_parts.append(X)

    X_all = np.vstack(X_parts)
    print(f"  [POOL] Training matrix: {X_all.shape} (subsampled to {PER_EVENT_CAP}/event)")

    # Column medians for NaN imputation
    col_medians = np.nanmedian(X_all, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    X_all = np.where(np.isnan(X_all), col_medians, X_all)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler().fit(X_all)
    X_scaled = np.clip(scaler.transform(X_all), -10, 10).astype(np.float32)

    return X_scaled, col_medians, scaler, shared_cols


def prepare_event_for_scoring(ev, shared_cols, col_medians, scaler):
    X = ev["enriched"][shared_cols].to_numpy(dtype=np.float32)
    X = np.where(np.isnan(X), col_medians, X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaler.transform(X), -10, 10).astype(np.float32)


def main():
    print("=" * 80)
    print("  ENSEMBLE SOTA v2 — CROSS-EVENT PROTOCOL")
    print("  Pool all NORMAL events as training, score ANOMALY events")
    print("=" * 80)

    events = pd.read_csv(EVENT_INFO, sep=";")
    t0 = time.time()

    # 1. Load all events with rich features (slow — FFT)
    print(f"\n[LOAD] Building rich features for {len(events)} events...")
    event_packs = []
    for i, (_, row) in enumerate(events.iterrows()):
        ep = load_event_features(row, use_fft=True)
        if ep is not None:
            event_packs.append(ep)
        print(f"  [{i+1}/{len(events)}] event_id={row['event_id']} "
              f"({row['event_label']}, fault={str(row.get('event_description',''))[:24]}) "
              f"{time.time()-t0:.0f}s elapsed")

    normal_events = [ep for ep in event_packs if ep["event_label"] == "normal"]
    anomaly_events = [ep for ep in event_packs if ep["event_label"] == "anomaly"]
    print(f"\n[SPLIT] {len(normal_events)} normal (train pool), "
          f"{len(anomaly_events)} anomaly (test)")

    # 2. Build pooled training set
    X_train_s, col_medians, scaler, shared_cols = build_pooled_training_set(normal_events)

    # 3. Train ensemble on pooled training data
    print(f"\n[TRAIN-VAE] {X_train_s.shape[0]:,} rows, {X_train_s.shape[1]} features...")
    t1 = time.time()
    vae_bundle = train_vae(X_train_s, n_epochs=40, patience=8,
                             hidden=VAE_HIDDEN, latent=VAE_LATENT, beta=0.1)
    print(f"  VAE trained in {time.time()-t1:.0f}s")

    print(f"[TRAIN-LSTM] seq_len={SEQ_LEN}...")
    t1 = time.time()
    lstm_bundle = train_lstm_ae(X_train_s, seq_len=SEQ_LEN,
                                  n_epochs=15, patience=4,
                                  hidden=LSTM_HIDDEN)
    print(f"  LSTM-AE trained in {time.time()-t1:.0f}s")

    print(f"[TRAIN-TRANSFORMER] seq_len={SEQ_LEN}...")
    t1 = time.time()
    tf_bundle = train_transformer_ae(X_train_s, seq_len=SEQ_LEN,
                                       n_epochs=10, patience=3,
                                       d_model=TRANSFORMER_DMODEL,
                                       n_heads=TRANSFORMER_HEADS)
    print(f"  Transformer-AE trained in {time.time()-t1:.0f}s")

    # Get training distribution scores for z-norm
    vae_train_scores = vae_anomaly_score(vae_bundle["model"],
                                          torch.FloatTensor(X_train_s))
    lstm_train_scores = lstm_ae_anomaly_score(lstm_bundle, X_train_s)
    tf_train_scores = transformer_ae_anomaly_score(tf_bundle, X_train_s)

    # 4. Score all events (both normal held-out and anomaly)
    print(f"\n[SCORE] Evaluating all {len(event_packs)} events...")
    results = []
    care_packets = []

    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<26} {'AUC-VAE':>8} "
          f"{'AUC-LSTM':>9} {'AUC-TF':>7} {'AUC-ENS':>8}")
    print("-" * 80)

    for ep in event_packs:
        X_s = prepare_event_for_scoring(ep, shared_cols, col_medians, scaler)

        vae_s = vae_anomaly_score(vae_bundle["model"], torch.FloatTensor(X_s))
        lstm_s = lstm_ae_anomaly_score(lstm_bundle, X_s)
        tf_s = transformer_ae_anomaly_score(tf_bundle, X_s)

        vae_z = z_normalize(vae_s, vae_train_scores)
        lstm_z = z_normalize(lstm_s, lstm_train_scores)
        tf_z = z_normalize(tf_s, tf_train_scores)

        ens = (vae_z + lstm_z + tf_z) / 3.0
        ens_smooth = ewma_smooth(ens, alpha=EWMA_ALPHA)

        y_true = ep["is_anomaly"]

        auc_vae = auc_lstm = auc_tf = auc_ens = None
        if ep["event_label"] == "anomaly" and y_true.sum() > 0 and y_true.sum() < len(y_true):
            auc_vae = round(float(roc_auc_score(y_true, vae_s)), 4)
            auc_lstm = round(float(roc_auc_score(y_true, lstm_s)), 4)
            auc_tf = round(float(roc_auc_score(y_true, tf_s)), 4)
            auc_ens = round(float(roc_auc_score(y_true, ens_smooth)), 4)

            print(f"{ep['event_id']:>4} {ep['event_label']:>8} "
                  f"{ep['event_description'][:24]:<26} "
                  f"{auc_vae:>8.4f} {auc_lstm:>9.4f} "
                  f"{auc_tf:>7.4f} {auc_ens:>8.4f}")

        # Threshold from training ensemble
        ens_train_z = (z_normalize(vae_train_scores, vae_train_scores) +
                        z_normalize(lstm_train_scores, lstm_train_scores) +
                        z_normalize(tf_train_scores, tf_train_scores)) / 3.0
        threshold = float(np.percentile(ens_train_z, 95))

        results.append({
            "event_id": ep["event_id"],
            "event_type": ep["event_label"],
            "event_description": ep["event_description"],
            "n_rows": ep["n_rows"],
            "n_anomaly_rows": int(y_true.sum()),
            "event_start_idx": ep["event_start_idx"],
            "event_end_idx": ep["event_end_idx"],
            "threshold": threshold,
            "auc_vae": auc_vae,
            "auc_lstm": auc_lstm,
            "auc_transformer": auc_tf,
            "auc_ensemble": auc_ens,
        })

        care_packets.append({
            "event_type": ep["event_label"],
            "y_true": y_true,
            "y_score": ens_smooth,
            "event_start_idx": ep["event_start_idx"],
            "event_end_idx": ep["event_end_idx"],
            "threshold": threshold,
        })

    # 5. Aggregate
    anomaly_results = [r for r in results if r["event_type"] == "anomaly"
                        and r["auc_ensemble"] is not None]

    print(f"\n{'='*80}")
    print(f"  AGGREGATE — Cross-Event Protocol + Ensemble")
    print(f"{'='*80}")

    if anomaly_results:
        for key in ["auc_vae", "auc_lstm", "auc_transformer", "auc_ensemble"]:
            vals = [r[key] for r in anomaly_results if r.get(key) is not None]
            print(f"  {key:<18} mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")

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
        "method": "Ensemble v2 cross-event protocol (pooled-normal training)",
        "seq_len": SEQ_LEN,
        "ewma_alpha": EWMA_ALPHA,
        "n_normal_events_train": len(normal_events),
        "n_anomaly_events_test": len(anomaly_results),
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
        "total_time_s": time.time() - t0,
    }
    with (RESULTS_DIR / "care_ensemble_v2.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[OK] Saved: {RESULTS_DIR / 'care_ensemble_v2.json'}")
    print(f"[TIME] {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
