"""
Ensemble SOTA v3 — CORRECT CARE protocol.

KEY FIX vs v2: use CARE's built-in `train_test` column and `status_type_id`.

Protocol (matches Nair et al. 2025, arxiv 2510.15010):
  - Training rows per event: train_test == 'train' AND status_type_id == 0
    (known pre-fault normal operation)
  - Test rows per event: train_test == 'prediction'
    (the anomaly window + margin; ~2-3k rows per event)
  - Labels on test rows: 1 if index in [event_start_id, event_end_id], else 0
  - Pooled training: concatenate training rows from ALL 22 events
  - Pooled test scoring: score each event's prediction region individually
  - Per-event AUC + POOLED AUC across all events
  - Weighted ensemble: weight each model by inverse val loss

Usage:
    python -m src.benchmark.ensemble_sota_v3
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

SEQ_LEN = 12
VAE_HIDDEN = 128
VAE_LATENT = 16
LSTM_HIDDEN = 64
TRANSFORMER_DMODEL = 64
TRANSFORMER_HEADS = 4
EWMA_ALPHA = 0.3


def z_normalize(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mu = float(reference.mean())
    sigma = float(reference.std())
    if sigma < 1e-9:
        return scores - mu
    return (scores - mu) / sigma


def load_event_with_splits(event_row, use_fft: bool = True):
    """Load one event's CSV AND return the train_test + status_type_id splits."""
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, sep=";", low_memory=False)

    # Grab train_test and status_type_id BEFORE renaming (in case they get dropped)
    train_test_raw = df["train_test"].values if "train_test" in df.columns else None
    status_type_raw = df["status_type_id"].values if "status_type_id" in df.columns else None

    enriched = build_event_features(df, use_fft=use_fft)
    eng_cols = engineered_feature_columns(enriched)

    from src.benchmark.schema import CANONICAL_RAW_FEATURES
    raw_avail = [c for c in CANONICAL_RAW_FEATURES
                  if c in enriched.columns and enriched[c].notna().sum() > 100]
    feature_cols = raw_avail + eng_cols

    n = len(enriched)

    # Build masks
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    if train_test_raw is not None:
        # Match length (enriched may drop rows?)
        m = min(len(train_test_raw), n)
        train_mask[:m] = (train_test_raw[:m] == "train")
        test_mask[:m] = (train_test_raw[:m] == "prediction")

    # Filter train to normal status only
    if status_type_raw is not None:
        m = min(len(status_type_raw), n)
        normal_mask = np.zeros(n, dtype=bool)
        normal_mask[:m] = (status_type_raw[:m] == 0)
        train_mask = train_mask & normal_mask

    # Label: 1 within [event_start_id, event_end_id]
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
        "train_mask": train_mask,
        "test_mask": test_mask,
        "n_rows": n,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def build_pooled_training_set(event_packs, per_event_cap: int = 15000):
    """Concatenate TRAINING rows (train_mask) across all events."""
    col_sets = [set(ep["feature_cols"]) for ep in event_packs if ep is not None]
    shared_cols = list(sorted(set.intersection(*col_sets))) if col_sets else []
    if not shared_cols:
        raise RuntimeError("No shared features")

    print(f"  [POOL] Shared features: {len(shared_cols)}")

    X_parts = []
    for ep in event_packs:
        if ep is None or ep["n_train"] == 0:
            continue
        X = ep["enriched"][shared_cols].to_numpy(dtype=np.float32)
        X_train_ev = X[ep["train_mask"]]
        if len(X_train_ev) > per_event_cap:
            idx = np.linspace(0, len(X_train_ev) - 1, per_event_cap).astype(int)
            X_train_ev = X_train_ev[idx]
        X_parts.append(X_train_ev)

    X_all = np.vstack(X_parts)
    print(f"  [POOL] Training matrix: {X_all.shape} "
          f"(subsampled to {per_event_cap}/event)")

    col_medians = np.nanmedian(X_all, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    X_all = np.where(np.isnan(X_all), col_medians, X_all)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler().fit(X_all)
    X_scaled = np.clip(scaler.transform(X_all), -10, 10).astype(np.float32)
    return X_scaled, col_medians, scaler, shared_cols


def prepare_event_features(ep, shared_cols, col_medians, scaler):
    X = ep["enriched"][shared_cols].to_numpy(dtype=np.float32)
    X = np.where(np.isnan(X), col_medians, X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaler.transform(X), -10, 10).astype(np.float32)


def main():
    print("=" * 80)
    print("  ENSEMBLE SOTA v3 — CORRECT CARE PROTOCOL")
    print("  Uses CARE's built-in train_test and status_type_id columns")
    print("=" * 80)

    events = pd.read_csv(EVENT_INFO, sep=";")
    t0 = time.time()

    print(f"\n[LOAD] Building rich features for {len(events)} events...")
    event_packs = []
    for i, (_, row) in enumerate(events.iterrows()):
        ep = load_event_with_splits(row, use_fft=True)
        if ep is not None:
            event_packs.append(ep)
            if (i + 1) % 5 == 0 or i < 3:
                print(f"  [{i+1}/{len(events)}] id={row['event_id']} "
                      f"({row['event_label'][:8]}) n_train={ep['n_train']:,} "
                      f"n_test={ep['n_test']:,} elapsed={time.time()-t0:.0f}s")

    print(f"\n[STATS] Loaded {len(event_packs)} events in {time.time()-t0:.0f}s")
    total_train = sum(ep["n_train"] for ep in event_packs)
    total_test = sum(ep["n_test"] for ep in event_packs)
    print(f"        Training rows available: {total_train:,}")
    print(f"        Test rows (prediction regions): {total_test:,}")

    X_train_s, col_medians, scaler, shared_cols = build_pooled_training_set(
        event_packs, per_event_cap=6000,
    )

    print(f"\n[TRAIN-VAE]...")
    t1 = time.time()
    vae_bundle = train_vae(X_train_s, n_epochs=40, patience=8,
                             hidden=VAE_HIDDEN, latent=VAE_LATENT, beta=0.1)
    vae_val = vae_bundle.get("best_val", 1.0)
    print(f"  VAE trained in {time.time()-t1:.0f}s (best_val={vae_val:.4f})")

    print(f"[TRAIN-LSTM] seq_len={SEQ_LEN}...")
    t1 = time.time()
    lstm_bundle = train_lstm_ae(X_train_s, seq_len=SEQ_LEN,
                                  n_epochs=15, patience=4, hidden=LSTM_HIDDEN)
    lstm_val = lstm_bundle.get("best_val", 1.0)
    print(f"  LSTM-AE trained in {time.time()-t1:.0f}s (best_val={lstm_val:.4f})")

    print(f"[TRAIN-TRANSFORMER] seq_len={SEQ_LEN}...")
    t1 = time.time()
    tf_bundle = train_transformer_ae(X_train_s, seq_len=SEQ_LEN,
                                       n_epochs=10, patience=3,
                                       d_model=TRANSFORMER_DMODEL,
                                       n_heads=TRANSFORMER_HEADS)
    tf_val = tf_bundle.get("best_val", 1.0)
    print(f"  Transformer-AE trained in {time.time()-t1:.0f}s (best_val={tf_val:.4f})")

    # Validation-weighted ensemble (inverse val loss)
    inv_vae = 1.0 / max(vae_val, 1e-6)
    inv_lstm = 1.0 / max(lstm_val, 1e-6)
    inv_tf = 1.0 / max(tf_val, 1e-6)
    total_w = inv_vae + inv_lstm + inv_tf
    w_vae = inv_vae / total_w
    w_lstm = inv_lstm / total_w
    w_tf = inv_tf / total_w
    print(f"\n[WEIGHTS] VAE={w_vae:.3f}, LSTM={w_lstm:.3f}, TF={w_tf:.3f}")

    # Training scores for z-norm reference
    vae_train = vae_anomaly_score(vae_bundle["model"], torch.FloatTensor(X_train_s))
    lstm_train = lstm_ae_anomaly_score(lstm_bundle, X_train_s)
    tf_train = transformer_ae_anomaly_score(tf_bundle, X_train_s)

    print(f"\n[SCORE] Per-event evaluation on prediction regions...")
    results = []
    care_packets = []
    pooled_scores = []
    pooled_labels = []

    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<26} {'n_test':>7} "
          f"{'AUC-VAE':>8} {'AUC-LSTM':>9} {'AUC-TF':>7} {'AUC-ENS':>8}")
    print("-" * 90)

    for ep in event_packs:
        X_s = prepare_event_features(ep, shared_cols, col_medians, scaler)

        vae_s = vae_anomaly_score(vae_bundle["model"], torch.FloatTensor(X_s))
        lstm_s = lstm_ae_anomaly_score(lstm_bundle, X_s)
        tf_s = transformer_ae_anomaly_score(tf_bundle, X_s)

        # Z-normalize each component
        vae_z = z_normalize(vae_s, vae_train)
        lstm_z = z_normalize(lstm_s, lstm_train)
        tf_z = z_normalize(tf_s, tf_train)

        # Weighted ensemble
        ens = w_vae * vae_z + w_lstm * lstm_z + w_tf * tf_z
        ens_smooth = ewma_smooth(ens, alpha=EWMA_ALPHA)

        y_true = ep["is_anomaly"]
        test_mask = ep["test_mask"]

        # Restrict AUC to test region (prediction split)
        y_test = y_true[test_mask]
        vae_test = vae_s[test_mask]
        lstm_test = lstm_s[test_mask]
        tf_test = tf_s[test_mask]
        ens_test = ens_smooth[test_mask]

        auc_vae = auc_lstm = auc_tf = auc_ens = None
        if len(y_test) > 0 and len(np.unique(y_test)) >= 2:
            auc_vae = round(float(roc_auc_score(y_test, vae_test)), 4)
            auc_lstm = round(float(roc_auc_score(y_test, lstm_test)), 4)
            auc_tf = round(float(roc_auc_score(y_test, tf_test)), 4)
            auc_ens = round(float(roc_auc_score(y_test, ens_test)), 4)

            print(f"{ep['event_id']:>4} {ep['event_label']:>8} "
                  f"{ep['event_description'][:24]:<26} "
                  f"{len(y_test):>7} "
                  f"{auc_vae:>8.4f} {auc_lstm:>9.4f} "
                  f"{auc_tf:>7.4f} {auc_ens:>8.4f}")

            # Accumulate for pooled AUC
            pooled_scores.append(ens_test)
            pooled_labels.append(y_test)

        ens_train_z = (w_vae * z_normalize(vae_train, vae_train) +
                        w_lstm * z_normalize(lstm_train, lstm_train) +
                        w_tf * z_normalize(tf_train, tf_train))
        threshold = float(np.percentile(ens_train_z, 95))

        results.append({
            "event_id": ep["event_id"],
            "event_type": ep["event_label"],
            "event_description": ep["event_description"],
            "n_rows": ep["n_rows"],
            "n_train": ep["n_train"],
            "n_test": ep["n_test"],
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

    # Aggregate
    anomaly_results = [r for r in results if r["event_type"] == "anomaly"
                        and r["auc_ensemble"] is not None]

    print(f"\n{'='*80}")
    print(f"  AGGREGATE — v3 (CARE train_test + status_type_id)")
    print(f"{'='*80}")

    if anomaly_results:
        for key in ["auc_vae", "auc_lstm", "auc_transformer", "auc_ensemble"]:
            vals = [r[key] for r in anomaly_results if r.get(key) is not None]
            print(f"  {key:<20} mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}")

        print(f"\n  Per-Fault-Type Ensemble AUC:")
        by_fault = {}
        for r in anomaly_results:
            by_fault.setdefault(r["event_description"], []).append(r["auc_ensemble"])
        for ft, aucs_ft in sorted(by_fault.items(), key=lambda kv: -np.mean(kv[1])):
            print(f"    {ft:<35} n={len(aucs_ft):>2}  "
                  f"mean={np.mean(aucs_ft):.4f}")

    # Pooled AUC across all anomaly events' prediction regions
    pooled_auc = None
    if pooled_scores:
        ps = np.concatenate(pooled_scores)
        pl = np.concatenate(pooled_labels)
        if len(np.unique(pl)) >= 2:
            pooled_auc = float(roc_auc_score(pl, ps))
            print(f"\n  POOLED AUC (concatenated test regions): {pooled_auc:.4f}")
            print(f"  Pooled test rows: {len(pl):,} (anomaly={pl.sum():,})")

    care = compute_care_score(care_packets)
    print(f"\n  CARE Score:")
    for k, v in care.items():
        print(f"    {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Ensemble v3 CORRECT PROTOCOL (train_test + status_type_id filter)",
        "seq_len": SEQ_LEN,
        "ewma_alpha": EWMA_ALPHA,
        "ensemble_weights": {"vae": w_vae, "lstm": w_lstm, "transformer": w_tf},
        "val_losses": {"vae": vae_val, "lstm": lstm_val, "transformer": tf_val},
        "mean_auc_ensemble": float(np.mean([r["auc_ensemble"] for r in anomaly_results]))
                               if anomaly_results else 0,
        "mean_auc_vae": float(np.mean([r["auc_vae"] for r in anomaly_results]))
                          if anomaly_results else 0,
        "mean_auc_lstm": float(np.mean([r["auc_lstm"] for r in anomaly_results]))
                           if anomaly_results else 0,
        "mean_auc_transformer": float(np.mean([r["auc_transformer"] for r in anomaly_results]))
                                  if anomaly_results else 0,
        "pooled_auc": pooled_auc,
        "care_score": care,
        "per_event": results,
        "total_time_s": time.time() - t0,
    }
    with (RESULTS_DIR / "care_ensemble_v3.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[OK] Saved: {RESULTS_DIR / 'care_ensemble_v3.json'}")
    print(f"[TIME] {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
