"""
Generic CARE farm ensemble evaluator (works for Farm A/B/C).

Protocol confirmed-SOTA-matching on Farm B:
  - Training: status_type_id == 0 AND train_test == 'train' (pooled across events)
  - Test:     train_test == 'prediction' (per-event)
  - Label:    status_type_id != 0 (the correct task — detect any non-normal state)
  - Model:    VAE + LSTM-AE + Transformer-AE, val-weighted ensemble
  - Metric:   Pooled AUC + per-event AUC

Outputs (per farm):
  docs/results/care_farm_{a,b,c}_ensemble.json
      Aggregate AUCs + per-event AUC summaries (status, event, precursor).
  docs/results/care_ensemble_v7_per_event_scores{,_b,_c}.json
      Per-row test-region anomaly scores + y_event/y_precursor/y_status/sf_mask
      per event_id. Schema matches the existing Farm-A file so
      pinn_stacker.py and pinn_stacker_crossfarm.py read both transparently.

Usage:
    python -m src.benchmark.ensemble_care_farm --farm A
    python -m src.benchmark.ensemble_care_farm --farm B
    python -m src.benchmark.ensemble_care_farm --farm C
"""

from pathlib import Path
import argparse
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

CARE_BASE = Path("data/real_scada/care/extracted")
RESULTS_DIR = Path("docs/results")
PRECURSOR_DIR = Path("data/benchmark/care_precursor")


def z_normalize(scores, ref):
    mu = float(ref.mean()); sigma = float(ref.std())
    if sigma < 1e-9:
        return scores - mu
    return (scores - mu) / sigma


def load_event(event_row, datasets_dir):
    event_id = int(event_row["event_id"])
    csv_path = datasets_dir / f"{event_id}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    n = len(df)

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


def build_pool(event_packs, per_event_cap):
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


def load_precursor_map(farm: str) -> dict:
    """Load CARE-Precursor labels for `farm`. Returns {event_id: row_dict}."""
    path = PRECURSOR_DIR / f"event_info_precursor_farm_{farm}.csv"
    if not path.exists():
        print(f"  [warn] no precursor labels at {path} — y_precursor will equal y_event")
        return {}
    df = pd.read_csv(path, sep=";")
    return {int(r["event_id"]): r for _, r in df.iterrows()}


def build_y_precursor(ep: dict, precursor_map: dict) -> np.ndarray:
    """Return a full-length y_precursor array for one event pack.

    For events with a logged precursor_start_id < event_start_id, the positive
    window is extended leftward to cover the precursor region. Otherwise this
    falls back to y_event identically.
    """
    y_event = ep["is_anomaly_event"]
    y_pre = y_event.copy()
    pre_row = precursor_map.get(int(ep["event_id"]))
    if pre_row is None:
        return y_pre
    ps_raw = pre_row.get("precursor_start_id")
    if ps_raw is None or pd.isna(ps_raw):
        return y_pre
    ps = int(ps_raw)
    ev_idx = np.where(y_event == 1)[0]
    if not len(ev_idx):
        return y_pre
    s, e = int(ev_idx[0]), int(ev_idx[-1])
    if 0 <= ps < s:
        y_pre[ps:e + 1] = 1
    return y_pre


def per_row_output_path(farm: str) -> Path:
    """Farm A keeps the historical filename so pinn_stacker.py finds it
    untouched; B / C get a farm suffix."""
    if farm == "A":
        return RESULTS_DIR / "care_ensemble_v7_per_event_scores.json"
    return RESULTS_DIR / f"care_ensemble_v7_per_event_scores_{farm.lower()}.json"


def run(farm: str, per_event_cap: int = 6000,
         vae_hidden: int = 256, vae_latent: int = 32,
         lstm_hidden: int = 96, tf_dmodel: int = 96, tf_heads: int = 4,
         seq_len: int = 12, ewma_alpha: float = 0.3):

    farm_dir = CARE_BASE / f"Wind Farm {farm}" / f"Wind Farm {farm}"
    datasets_dir = farm_dir / "datasets"
    event_info = farm_dir / "event_info.csv"
    out_path = RESULTS_DIR / f"care_farm_{farm.lower()}_ensemble.json"
    per_row_path = per_row_output_path(farm)

    print("=" * 80)
    print(f"  ENSEMBLE Farm {farm} — status_type_id label + per-row dump")
    print("=" * 80)

    events = pd.read_csv(event_info, sep=";")
    precursor_map = load_precursor_map(farm)
    if precursor_map:
        print(f"  Loaded {len(precursor_map)} precursor entries from "
              f"event_info_precursor_farm_{farm}.csv")
    t0 = time.time()

    print(f"\n[LOAD] {len(events)} events...")
    event_packs = []
    for i, (_, row) in enumerate(events.iterrows()):
        ep = load_event(row, datasets_dir)
        if ep is not None:
            event_packs.append(ep)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(events)}] elapsed={time.time()-t0:.0f}s")

    print(f"\n[STATS] {len(event_packs)} events, "
          f"train rows: {sum(ep['n_train'] for ep in event_packs):,}, "
          f"test rows: {sum(ep['n_test'] for ep in event_packs):,}")

    X_train_s, col_medians, scaler, shared_cols = build_pool(event_packs, per_event_cap)

    print(f"\n[TRAIN-VAE]...")
    t1 = time.time()
    vae_bundle = train_vae(X_train_s, n_epochs=40, patience=8,
                             hidden=vae_hidden, latent=vae_latent, beta=0.1)
    print(f"  VAE {time.time()-t1:.0f}s val={vae_bundle['best_val']:.4f}")

    print(f"[TRAIN-LSTM]...")
    t1 = time.time()
    lstm_bundle = train_lstm_ae(X_train_s, seq_len=seq_len, n_epochs=15,
                                  patience=4, hidden=lstm_hidden)
    print(f"  LSTM {time.time()-t1:.0f}s val={lstm_bundle['best_val']:.4f}")

    print(f"[TRAIN-TRANSFORMER]...")
    t1 = time.time()
    tf_bundle = train_transformer_ae(X_train_s, seq_len=seq_len, n_epochs=10,
                                       patience=3, d_model=tf_dmodel,
                                       n_heads=tf_heads)
    print(f"  TF {time.time()-t1:.0f}s val={tf_bundle['best_val']:.4f}")

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
    pooled_labels_precursor = []
    per_row_dump: dict[int, dict] = {}

    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<28} {'n_test':>6} "
          f"{'AUC-st':>8} {'AUC-ev':>8} {'AUC-pre':>8}")
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
        ens_s = ewma_smooth(ens, alpha=ewma_alpha)

        test_mask = ep["test_mask"]
        y_event_full = ep["is_anomaly_event"]
        y_precursor_full = build_y_precursor(ep, precursor_map)

        y_st = ep["is_anomaly_status"][test_mask]
        y_ev = y_event_full[test_mask]
        y_pre = y_precursor_full[test_mask]
        ens_t = ens_s[test_mask]

        auc_st = auc_ev = auc_pre = None
        if len(np.unique(y_st)) == 2:
            auc_st = float(roc_auc_score(y_st, ens_t))
        if len(np.unique(y_ev)) == 2:
            auc_ev = float(roc_auc_score(y_ev, ens_t))
        if len(np.unique(y_pre)) == 2:
            auc_pre = float(roc_auc_score(y_pre, ens_t))

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  —   "
        print(f"{ep['event_id']:>4} {ep['event_label']:>8} "
              f"{ep['event_description'][:26]:<28} {len(y_st):>6} "
              f"{fmt(auc_st):>8} {fmt(auc_ev):>8} {fmt(auc_pre):>8}")

        pooled_scores.append(ens_t)
        pooled_labels_status.append(y_st)
        pooled_labels_event.append(y_ev)
        pooled_labels_precursor.append(y_pre)

        results.append({
            "event_id": ep["event_id"],
            "event_type": ep["event_label"],
            "event_description": ep["event_description"],
            "n_test": ep["n_test"],
            "auc_status": auc_st,
            "auc_event": auc_ev,
            "auc_precursor": auc_pre,
        })

        # Per-row dump in the schema the rest of the pipeline expects
        per_row_dump[int(ep["event_id"])] = {
            "scores":      [float(x) for x in ens_t],
            "y_event":     [int(x) for x in y_ev],
            "y_precursor": [int(x) for x in y_pre],
            "y_status":    [int(x) for x in y_st],
            "sf_mask":     [True] * int(test_mask.sum()),
        }

    print(f"\n{'='*80}")
    print(f"  AGGREGATE — Farm {farm}")
    print(f"{'='*80}")
    for key in ["auc_status", "auc_event", "auc_precursor"]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"  {key:<20} mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}  n={len(vals)}")

    ps = np.concatenate(pooled_scores)
    pls = np.concatenate(pooled_labels_status)
    ple = np.concatenate(pooled_labels_event)
    plp = np.concatenate(pooled_labels_precursor)
    pooled_st = float(roc_auc_score(pls, ps)) if len(np.unique(pls)) == 2 else None
    pooled_ev = float(roc_auc_score(ple, ps)) if len(np.unique(ple)) == 2 else None
    pooled_pre = float(roc_auc_score(plp, ps)) if len(np.unique(plp)) == 2 else None
    print(f"\n  POOLED AUC (status label):    {pooled_st if pooled_st else 'n/a'}")
    print(f"  POOLED AUC (event label):     {pooled_ev if pooled_ev else 'n/a'}")
    print(f"  POOLED AUC (precursor label): {pooled_pre if pooled_pre else 'n/a'}")
    print(f"  Pooled test rows: {len(pls):,} "
          f"(status=1: {pls.sum():,}, event=1: {ple.sum():,}, "
          f"precursor=1: {plp.sum():,})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": f"Ensemble Farm {farm}, status_type_id label",
        "n_features": len(shared_cols),
        "n_events": len(event_packs),
        "ensemble_weights": {"vae": float(w[0]), "lstm": float(w[1]),
                              "transformer": float(w[2])},
        "mean_auc_status": float(np.mean([r["auc_status"] for r in results
                                             if r.get("auc_status") is not None])),
        "mean_auc_event": (float(np.mean([r["auc_event"] for r in results
                                            if r.get("auc_event") is not None]))
                            if any(r.get("auc_event") is not None for r in results) else None),
        "mean_auc_precursor": (float(np.mean([r["auc_precursor"] for r in results
                                                if r.get("auc_precursor") is not None]))
                                if any(r.get("auc_precursor") is not None for r in results) else None),
        "pooled_auc_status": pooled_st,
        "pooled_auc_event": pooled_ev,
        "pooled_auc_precursor": pooled_pre,
        "per_event": results,
        "total_time_s": time.time() - t0,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Saved aggregate: {out_path}")

    # Per-row dump (string-keyed for JSON, matches Farm-A schema)
    with per_row_path.open("w") as f:
        json.dump({str(k): v for k, v in per_row_dump.items()}, f, default=str)
    print(f"[OK] Saved per-row scores: {per_row_path}")
    print(f"[TIME] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", choices=["A", "B", "C"], default="C")
    parser.add_argument("--per-event-cap", type=int, default=6000)
    args = parser.parse_args()
    run(args.farm, per_event_cap=args.per_event_cap)
