"""
Ensemble SOTA evaluator for CARE Farm B (SOTA reports AUC 0.947 here).

Farm B has:
  - 15 events (6 anomaly + 9 normal)
  - 257 anonymized features per sensor: avg, min, max, std already computed
  - train_test + status_type_id columns (same protocol as Farm A)

We use ALL sensor_*_avg columns as base features + the pre-computed stats.
No FFT / rolling window — Farm B already provides per-10min statistics.
Label: status_type_id != 0.

Usage:
    python -m src.benchmark.ensemble_farm_b
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

from src.benchmark.vae_baseline import train_vae, vae_anomaly_score
from src.benchmark.lstm_ae import train_lstm_ae, lstm_ae_anomaly_score
from src.benchmark.transformer_ae import train_transformer_ae, transformer_ae_anomaly_score
from src.benchmark.care_sota import ewma_smooth

CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm B/Wind Farm B")
DATASETS_DIR = CARE_DIR / "datasets"
EVENT_INFO = CARE_DIR / "event_info.csv"
RESULTS_DIR = Path("docs/results")

SEQ_LEN = 12
VAE_HIDDEN = 256
VAE_LATENT = 32
LSTM_HIDDEN = 96
TRANSFORMER_DMODEL = 96
TRANSFORMER_HEADS = 4
EWMA_ALPHA = 0.3


def z_normalize(scores, ref):
    mu = float(ref.mean()); sigma = float(ref.std())
    if sigma < 1e-9:
        return scores - mu
    return (scores - mu) / sigma


def load_event(event_row):
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    n = len(df)

    # Numeric sensor columns (skip id/timestamp/status/train_test)
    skip = {"time_stamp", "asset_id", "id", "train_test", "status_type_id"}
    feature_cols = [c for c in df.columns if c not in skip
                    and pd.api.types.is_numeric_dtype(df[c])]

    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    if "train_test" in df.columns:
        train_mask = (df["train_test"].values == "train")
        test_mask = (df["train_test"].values == "prediction")

    status_arr = np.zeros(n, dtype=int)
    if "status_type_id" in df.columns:
        status_arr = np.nan_to_num(df["status_type_id"].values, nan=0).astype(int)
    train_mask = train_mask & (status_arr == 0)

    is_anomaly_status = (status_arr != 0).astype(int)
    is_anomaly_event = np.zeros(n, dtype=int)
    if event_row["event_label"] == "anomaly":
        s = int(event_row["event_start_id"])
        e = int(event_row["event_end_id"])
        is_anomaly_event[s:e + 1] = 1

    return {
        "event_id": event_id,
        "event_label": event_row["event_label"],
        "event_description": str(event_row.get("event_description", ""))[:40],
        "df": df,
        "feature_cols": feature_cols,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "is_anomaly_status": is_anomaly_status,
        "is_anomaly_event": is_anomaly_event,
        "n_rows": n,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def build_pool(event_packs, per_event_cap=8000):
    col_sets = [set(ep["feature_cols"]) for ep in event_packs]
    shared_cols = sorted(set.intersection(*col_sets))
    print(f"  [POOL] Shared features: {len(shared_cols)}")

    X_parts = []
    for ep in event_packs:
        if ep["n_train"] == 0:
            continue
        X = ep["df"][shared_cols].to_numpy(dtype=np.float32)
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
    X = ep["df"][shared_cols].to_numpy(dtype=np.float32)
    X = np.where(np.isnan(X), col_medians, X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaler.transform(X), -10, 10).astype(np.float32)


def main():
    print("=" * 80)
    print("  ENSEMBLE Farm B — status_type_id label")
    print("=" * 80)

    events = pd.read_csv(EVENT_INFO, sep=";")
    t0 = time.time()

    print(f"\n[LOAD] {len(events)} events...")
    event_packs = []
    for i, (_, row) in enumerate(events.iterrows()):
        ep = load_event(row)
        if ep is not None:
            event_packs.append(ep)
            print(f"  [{i+1}/{len(events)}] id={row['event_id']} "
                  f"({row['event_label']}) {ep['n_train']:,}/{ep['n_test']:,} "
                  f"elapsed={time.time()-t0:.0f}s")

    print(f"\n[STATS] {len(event_packs)} events loaded")
    print(f"        Total train rows: {sum(ep['n_train'] for ep in event_packs):,}")
    print(f"        Total test rows: {sum(ep['n_test'] for ep in event_packs):,}")

    X_train_s, col_medians, scaler, shared_cols = build_pool(event_packs)

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

    # Weighted ensemble (scale-adjusted)
    n_feat = X_train_s.shape[1]
    vae_val_adj = vae_bundle["best_val"] / n_feat
    inv = np.array([1.0 / max(vae_val_adj, 1e-6),
                     1.0 / max(lstm_bundle["best_val"], 1e-6),
                     1.0 / max(tf_bundle["best_val"], 1e-6)])
    w = inv / inv.sum()
    print(f"\n[WEIGHTS] VAE={w[0]:.3f} LSTM={w[1]:.3f} TF={w[2]:.3f}")

    vae_train = vae_anomaly_score(vae_bundle["model"], torch.FloatTensor(X_train_s))
    lstm_train = lstm_ae_anomaly_score(lstm_bundle, X_train_s)
    tf_train = transformer_ae_anomaly_score(tf_bundle, X_train_s)

    print(f"\n[SCORE]...")
    results = []
    pooled_scores = []
    pooled_labels_status = []
    pooled_labels_event = []

    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<32} {'n_test':>6} "
          f"{'ENS-st':>7} {'ENS-ev':>7}")
    print("-" * 78)

    for ep in event_packs:
        X_s = prepare_event(ep, shared_cols, col_medians, scaler)
        vae_s = vae_anomaly_score(vae_bundle["model"], torch.FloatTensor(X_s))
        lstm_s = lstm_ae_anomaly_score(lstm_bundle, X_s)
        tf_s = transformer_ae_anomaly_score(tf_bundle, X_s)

        vae_z = z_normalize(vae_s, vae_train)
        lstm_z = z_normalize(lstm_s, lstm_train)
        tf_z = z_normalize(tf_s, tf_train)
        ens = w[0] * vae_z + w[1] * lstm_z + w[2] * tf_z
        ens_s = ewma_smooth(ens, alpha=EWMA_ALPHA)

        test_mask = ep["test_mask"]
        y_st = ep["is_anomaly_status"][test_mask]
        y_ev = ep["is_anomaly_event"][test_mask]
        ens_t = ens_s[test_mask]

        auc_st = auc_ev = None
        if len(np.unique(y_st)) == 2:
            auc_st = float(roc_auc_score(y_st, ens_t))
        if len(np.unique(y_ev)) == 2:
            auc_ev = float(roc_auc_score(y_ev, ens_t))

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  —   "
        print(f"{ep['event_id']:>4} {ep['event_label']:>8} "
              f"{ep['event_description'][:30]:<32} {len(y_st):>6} "
              f"{fmt(auc_st):>7} {fmt(auc_ev):>7}")

        pooled_scores.append(ens_t)
        pooled_labels_status.append(y_st)
        pooled_labels_event.append(y_ev)

        results.append({
            "event_id": ep["event_id"],
            "event_type": ep["event_label"],
            "event_description": ep["event_description"],
            "n_test": ep["n_test"],
            "auc_status": auc_st,
            "auc_event": auc_ev,
        })

    print(f"\n{'='*80}")
    print(f"  AGGREGATE — Farm B (257 features, status_type_id label)")
    print(f"{'='*80}")

    for key in ["auc_status", "auc_event"]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"  {key:<20} mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}  n={len(vals)}")

    ps = np.concatenate(pooled_scores)
    pls = np.concatenate(pooled_labels_status)
    ple = np.concatenate(pooled_labels_event)
    print(f"\n  POOLED AUC (status label): {roc_auc_score(pls, ps):.4f}")
    print(f"  POOLED AUC (event label):  {roc_auc_score(ple, ps):.4f}")
    print(f"  Pooled test rows: {len(pls):,} "
          f"(status=1: {pls.sum():,}, event=1: {ple.sum():,})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Ensemble Farm B, status_type_id label",
        "n_features": len(shared_cols),
        "mean_auc_status": float(np.mean([r["auc_status"] for r in results
                                             if r.get("auc_status") is not None])),
        "mean_auc_event": float(np.mean([r["auc_event"] for r in results
                                            if r.get("auc_event") is not None]))
                            if any(r.get("auc_event") for r in results) else None,
        "pooled_auc_status": float(roc_auc_score(pls, ps)),
        "pooled_auc_event": float(roc_auc_score(ple, ps)),
        "per_event": results,
        "total_time_s": time.time() - t0,
    }
    with (RESULTS_DIR / "care_farm_b_ensemble.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Saved: {RESULTS_DIR / 'care_farm_b_ensemble.json'}")
    print(f"[TIME] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
