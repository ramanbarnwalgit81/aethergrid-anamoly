"""
Ensemble SOTA v4 — LABEL FIX + per-turbine normalization.

Hypothesis: SOTA paper uses `status_type_id != 0` as the anomaly label
(any non-normal turbine state), not `event_start_id:event_end_id`
(only the documented fault window).

This changes the task from "detect the documented fault window" to
"detect any non-normal operational state", which matches typical wind
CMS deployments (status codes mark equipment-originating alerts).

Also adds per-event (effectively per-turbine) normalization of scores
before AUC: z-score within each event then pool.

Protocol:
  - Training rows: train_test == 'train' AND status_type_id == 0 (pure normal)
  - Test rows:     train_test == 'prediction' (mixed states)
  - Anomaly label: status_type_id != 0 on test rows
  - Evaluate: per-event AUC + POOLED AUC on all test rows

Usage:
    python -m src.benchmark.ensemble_sota_v4
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


def z_normalize(scores, ref):
    mu = float(ref.mean())
    sigma = float(ref.std())
    if sigma < 1e-9:
        return scores - mu
    return (scores - mu) / sigma


def load_event(event_row, use_fft=True):
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    train_test_raw = df["train_test"].values if "train_test" in df.columns else None
    status_type_raw = df["status_type_id"].values if "status_type_id" in df.columns else None

    enriched = build_event_features(df, use_fft=use_fft)
    eng_cols = engineered_feature_columns(enriched)
    from src.benchmark.schema import CANONICAL_RAW_FEATURES
    raw_avail = [c for c in CANONICAL_RAW_FEATURES
                  if c in enriched.columns and enriched[c].notna().sum() > 100]
    feature_cols = raw_avail + eng_cols

    n = len(enriched)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    status_arr = np.zeros(n, dtype=int)
    if train_test_raw is not None:
        m = min(len(train_test_raw), n)
        train_mask[:m] = (train_test_raw[:m] == "train")
        test_mask[:m] = (train_test_raw[:m] == "prediction")
    if status_type_raw is not None:
        m = min(len(status_type_raw), n)
        status_arr[:m] = np.nan_to_num(status_type_raw[:m], nan=0).astype(int)
    train_mask = train_mask & (status_arr == 0)

    # NEW LABEL: status_type_id != 0 = anomaly
    is_anomaly_status = (status_arr != 0).astype(int)
    # OLD LABEL: event window
    is_anomaly_event = np.zeros(n, dtype=int)
    if event_row["event_label"] == "anomaly":
        s = int(event_row["event_start_id"])
        e = int(event_row["event_end_id"])
        is_anomaly_event[s:e + 1] = 1

    return {
        "event_id": event_id,
        "event_label": event_row["event_label"],
        "event_description": str(event_row.get("event_description", "")),
        "enriched": enriched,
        "feature_cols": feature_cols,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "is_anomaly_status": is_anomaly_status,
        "is_anomaly_event": is_anomaly_event,
        "n_rows": n,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def build_pool(event_packs, per_event_cap=6000):
    col_sets = [set(ep["feature_cols"]) for ep in event_packs]
    shared_cols = sorted(set.intersection(*col_sets))
    print(f"  [POOL] Shared features: {len(shared_cols)}")

    X_parts = []
    for ep in event_packs:
        if ep["n_train"] == 0:
            continue
        X = ep["enriched"][shared_cols].to_numpy(dtype=np.float32)
        X_tr = X[ep["train_mask"]]
        if len(X_tr) > per_event_cap:
            idx = np.linspace(0, len(X_tr) - 1, per_event_cap).astype(int)
            X_tr = X_tr[idx]
        X_parts.append(X_tr)

    X_all = np.vstack(X_parts)
    col_medians = np.nanmedian(X_all, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    X_all = np.where(np.isnan(X_all), col_medians, X_all)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = RobustScaler().fit(X_all)
    X_scaled = np.clip(scaler.transform(X_all), -10, 10).astype(np.float32)
    print(f"  [POOL] Training matrix: {X_scaled.shape}")
    return X_scaled, col_medians, scaler, shared_cols


def prepare_event(ep, shared_cols, col_medians, scaler):
    X = ep["enriched"][shared_cols].to_numpy(dtype=np.float32)
    X = np.where(np.isnan(X), col_medians, X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaler.transform(X), -10, 10).astype(np.float32)


def main():
    print("=" * 80)
    print("  ENSEMBLE SOTA v4 — status_type_id label + per-event normalization")
    print("=" * 80)

    events = pd.read_csv(EVENT_INFO, sep=";")
    t0 = time.time()

    print(f"\n[LOAD] {len(events)} events...")
    event_packs = []
    for i, (_, row) in enumerate(events.iterrows()):
        ep = load_event(row)
        if ep is not None:
            event_packs.append(ep)
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(events)}] elapsed={time.time()-t0:.0f}s")

    print(f"\n[STATS] Loaded {len(event_packs)} events in {time.time()-t0:.0f}s")
    print(f"        Total train rows (status=0): {sum(ep['n_train'] for ep in event_packs):,}")
    print(f"        Total test rows: {sum(ep['n_test'] for ep in event_packs):,}")

    X_train_s, col_medians, scaler, shared_cols = build_pool(event_packs, per_event_cap=6000)

    print(f"\n[TRAIN-VAE]...")
    t1 = time.time()
    vae_bundle = train_vae(X_train_s, n_epochs=40, patience=8,
                             hidden=VAE_HIDDEN, latent=VAE_LATENT, beta=0.1)
    print(f"  VAE {time.time()-t1:.0f}s val={vae_bundle['best_val']:.4f}")

    print(f"[TRAIN-LSTM]...")
    t1 = time.time()
    lstm_bundle = train_lstm_ae(X_train_s, seq_len=SEQ_LEN, n_epochs=15,
                                  patience=4, hidden=LSTM_HIDDEN)
    print(f"  LSTM {time.time()-t1:.0f}s val={lstm_bundle['best_val']:.4f}")

    print(f"[TRAIN-TRANSFORMER]...")
    t1 = time.time()
    tf_bundle = train_transformer_ae(X_train_s, seq_len=SEQ_LEN, n_epochs=10,
                                       patience=3, d_model=TRANSFORMER_DMODEL,
                                       n_heads=TRANSFORMER_HEADS)
    print(f"  TF {time.time()-t1:.0f}s val={tf_bundle['best_val']:.4f}")

    # Validation-weighted ensemble (per-feature MSE scale adjustment for VAE)
    # VAE val_loss is MSE SUM so divide by n_features to compare fairly
    n_feat = X_train_s.shape[1]
    vae_val_adj = vae_bundle["best_val"] / n_feat
    lstm_val = lstm_bundle["best_val"]
    tf_val = tf_bundle["best_val"]
    print(f"\n[WEIGHTS] scale-adjusted: VAE={vae_val_adj:.4f} LSTM={lstm_val:.4f} TF={tf_val:.4f}")
    inv = np.array([1.0 / max(vae_val_adj, 1e-6), 1.0 / max(lstm_val, 1e-6),
                     1.0 / max(tf_val, 1e-6)])
    w = inv / inv.sum()
    print(f"[WEIGHTS] final: VAE={w[0]:.3f} LSTM={w[1]:.3f} TF={w[2]:.3f}")

    vae_train = vae_anomaly_score(vae_bundle["model"], torch.FloatTensor(X_train_s))
    lstm_train = lstm_ae_anomaly_score(lstm_bundle, X_train_s)
    tf_train = transformer_ae_anomaly_score(tf_bundle, X_train_s)

    print(f"\n[SCORE] Per-event evaluation with BOTH labelings...")
    results = []
    pooled_scores = []
    pooled_labels_status = []
    pooled_labels_event = []

    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<24} {'n_test':>6} "
          f"{'ENS-st':>7} {'ENS-ev':>7} {'TF-st':>7} {'TF-ev':>7}")
    print("-" * 90)

    for ep in event_packs:
        X_s = prepare_event(ep, shared_cols, col_medians, scaler)

        vae_s = vae_anomaly_score(vae_bundle["model"], torch.FloatTensor(X_s))
        lstm_s = lstm_ae_anomaly_score(lstm_bundle, X_s)
        tf_s = transformer_ae_anomaly_score(tf_bundle, X_s)

        vae_z = z_normalize(vae_s, vae_train)
        lstm_z = z_normalize(lstm_s, lstm_train)
        tf_z = z_normalize(tf_s, tf_train)

        ens = w[0] * vae_z + w[1] * lstm_z + w[2] * tf_z
        ens_smooth = ewma_smooth(ens, alpha=EWMA_ALPHA)

        test_mask = ep["test_mask"]
        y_status = ep["is_anomaly_status"][test_mask]
        y_event = ep["is_anomaly_event"][test_mask]
        ens_t = ens_smooth[test_mask]
        tf_t = tf_s[test_mask]

        auc_ens_st = auc_ens_ev = auc_tf_st = auc_tf_ev = None
        if len(y_status) >= 2 and len(np.unique(y_status)) == 2:
            auc_ens_st = float(roc_auc_score(y_status, ens_t))
            auc_tf_st = float(roc_auc_score(y_status, tf_t))
        if len(y_event) >= 2 and len(np.unique(y_event)) == 2:
            auc_ens_ev = float(roc_auc_score(y_event, ens_t))
            auc_tf_ev = float(roc_auc_score(y_event, tf_t))

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  —   "

        print(f"{ep['event_id']:>4} {ep['event_label']:>8} "
              f"{ep['event_description'][:22]:<24} {len(y_status):>6} "
              f"{fmt(auc_ens_st):>7} {fmt(auc_ens_ev):>7} "
              f"{fmt(auc_tf_st):>7} {fmt(auc_tf_ev):>7}")

        pooled_scores.append(ens_t)
        pooled_labels_status.append(y_status)
        pooled_labels_event.append(y_event)

        results.append({
            "event_id": ep["event_id"],
            "event_type": ep["event_label"],
            "event_description": ep["event_description"],
            "n_test": ep["n_test"],
            "auc_ens_status_label": auc_ens_st,
            "auc_ens_event_label": auc_ens_ev,
            "auc_tf_status_label": auc_tf_st,
            "auc_tf_event_label": auc_tf_ev,
        })

    print(f"\n{'='*80}")
    print(f"  AGGREGATE — v4")
    print(f"{'='*80}")

    for key in ["auc_ens_status_label", "auc_ens_event_label",
                  "auc_tf_status_label", "auc_tf_event_label"]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"  {key:<30} mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}  n={len(vals)}")

    # Pooled
    ps = np.concatenate(pooled_scores)
    pls = np.concatenate(pooled_labels_status)
    ple = np.concatenate(pooled_labels_event)
    print(f"\n  POOLED AUC (status_type_id != 0 label): "
          f"{roc_auc_score(pls, ps):.4f}")
    print(f"  POOLED AUC (event window label):        "
          f"{roc_auc_score(ple, ps):.4f}")
    print(f"  Pooled test rows: {len(pls):,}  "
          f"(status=1: {pls.sum():,}, event=1: {ple.sum():,})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Ensemble v4 — dual labeling (status_type_id + event window)",
        "mean_auc_ens_status": float(np.mean([r["auc_ens_status_label"] for r in results
                                                if r.get("auc_ens_status_label") is not None])),
        "mean_auc_ens_event": float(np.mean([r["auc_ens_event_label"] for r in results
                                               if r.get("auc_ens_event_label") is not None])),
        "pooled_auc_status": float(roc_auc_score(pls, ps)),
        "pooled_auc_event": float(roc_auc_score(ple, ps)),
        "per_event": results,
        "total_time_s": time.time() - t0,
    }
    with (RESULTS_DIR / "care_ensemble_v4.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Saved: {RESULTS_DIR / 'care_ensemble_v4.json'}")
    print(f"[TIME] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
