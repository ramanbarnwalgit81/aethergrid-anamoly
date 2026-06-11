"""
Ensemble SOTA v6 — Fleet-NBM residual features + ensemble (hard task focus).

PROTOCOL
--------
v6 = v3 + Fleet-NBM residuals as additional features.

Fleet-NBM trained on Penmanshiel (14 turbines × 4.5 years, 3.3 M rows
of healthy-operation data) predicts each of 4 target signals
(main_bearing_temp, generator_bearing_de, generator_bearing_nde,
nacelle_temp) given 7 conditioning variables.

For each CARE event row, we compute:
  - r_i(t)     = actual_i(t) - predicted_i(t)   per target i
  - |z_r_i(t)| = robust z-score of |r_i(t)|
  - rolling_24h_mean(|z_r_i|), rolling_24h_std(|z_r_i|)
  - max_i(|z_r_i|), sum_i(|z_r_i|)

These become additional columns in the feature matrix that goes into
the VAE+LSTM+TF ensemble. The ensemble now has FLEET-RELATIVE SIGNALS
that single-turbine training cannot have learned.

Labels: dual (status AND event-window) like v4/v5.

Usage:
    python -m src.benchmark.ensemble_sota_v6
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
from src.benchmark.fleet_nbm import (
    load_nbm, compute_nbm_residual, TARGET_COLS as NBM_TARGETS,
)


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
    mu = float(ref.mean()); sigma = float(ref.std())
    if sigma < 1e-9:
        return scores - mu
    return (scores - mu) / sigma


def compute_fleet_residual_features(df: pd.DataFrame, nbms: dict) -> dict:
    """
    Compute Fleet-NBM residual features for one event's dataframe.
    df should already have canonical column names (after build_event_features).

    Returns a dict of column_name -> np.ndarray of length len(df).
    """
    n = len(df)
    features = {}

    for target, bundle in nbms.items():
        if target not in df.columns:
            continue
        r = compute_nbm_residual(df, bundle)

        # Robust z-score
        med = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - med)) + 1e-6
        z = (r - med) / (1.4826 * mad)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        abs_z = np.abs(z)

        features[f"fleet_nbm_{target}_residual"] = r.astype(np.float32)
        features[f"fleet_nbm_{target}_abs_z"] = abs_z.astype(np.float32)

        # Rolling 24-row (≈4 hr at 10-min res) mean and std of |z|
        s = pd.Series(abs_z)
        features[f"fleet_nbm_{target}_absz_ma24"] = \
            s.rolling(24, min_periods=3).mean().fillna(0).to_numpy(dtype=np.float32)
        features[f"fleet_nbm_{target}_absz_ma144"] = \
            s.rolling(144, min_periods=6).mean().fillna(0).to_numpy(dtype=np.float32)  # ~24 hr
        features[f"fleet_nbm_{target}_absz_std24"] = \
            s.rolling(24, min_periods=3).std().fillna(0).to_numpy(dtype=np.float32)

    # Aggregation features across targets
    all_absz = np.stack(
        [features[c] for c in features if c.endswith("_abs_z")], axis=1,
    ) if any(c.endswith("_abs_z") for c in features) else None

    if all_absz is not None and all_absz.shape[1] > 0:
        features["fleet_nbm_max_abs_z"] = np.nanmax(all_absz, axis=1).astype(np.float32)
        features["fleet_nbm_sum_abs_z"] = np.nansum(all_absz, axis=1).astype(np.float32)
        features["fleet_nbm_mean_abs_z"] = np.nanmean(all_absz, axis=1).astype(np.float32)

    return features


def load_event_with_nbm(event_row, nbms: dict, use_fft: bool = True):
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return None

    df_raw = pd.read_csv(csv_path, sep=";", low_memory=False)
    train_test_raw = df_raw["train_test"].values if "train_test" in df_raw.columns else None
    status_type_raw = df_raw["status_type_id"].values if "status_type_id" in df_raw.columns else None

    enriched = build_event_features(df_raw, use_fft=use_fft)

    # Add Fleet-NBM residual features
    fleet_features = compute_fleet_residual_features(enriched, nbms)
    for col, vals in fleet_features.items():
        # Align length (enriched may have had a few rows dropped)
        m = min(len(enriched), len(vals))
        enriched[col] = np.nan
        enriched.loc[enriched.index[:m], col] = vals[:m]

    eng_cols = engineered_feature_columns(enriched)
    fleet_cols = [c for c in enriched.columns if c.startswith("fleet_nbm_")]

    from src.benchmark.schema import CANONICAL_RAW_FEATURES
    raw_avail = [c for c in CANONICAL_RAW_FEATURES
                  if c in enriched.columns and enriched[c].notna().sum() > 100]
    feature_cols = raw_avail + eng_cols + fleet_cols

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
        "enriched": enriched,
        "feature_cols": feature_cols,
        "fleet_cols": fleet_cols,
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
    fleet_in_shared = [c for c in shared_cols if c.startswith("fleet_nbm_")]
    print(f"  [POOL] Shared features: {len(shared_cols)} "
          f"(including {len(fleet_in_shared)} fleet-NBM)")

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
    print("  ENSEMBLE SOTA v6 — Fleet-NBM residuals as ensemble features")
    print("  Hard task focus: push CARE Farm A event-window AUC past 0.60")
    print("=" * 80)

    # Load Fleet-NBMs
    nbms = {}
    for target in NBM_TARGETS:
        b = load_nbm("penmanshiel", target)
        if b is not None:
            nbms[target] = b
    print(f"[NBM] Loaded {len(nbms)} Fleet-NBMs: {list(nbms.keys())}")
    if not nbms:
        print("[NBM] ERROR — no NBMs loaded. Run `fleet_nbm train --source penmanshiel` first.")
        return

    events = pd.read_csv(EVENT_INFO, sep=";")
    t0 = time.time()

    print(f"\n[LOAD] Building rich+fleet features for {len(events)} events...")
    event_packs = []
    for i, (_, row) in enumerate(events.iterrows()):
        ep = load_event_with_nbm(row, nbms)
        if ep is not None:
            event_packs.append(ep)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(events)}] id={row['event_id']} "
                  f"({row['event_label']}) elapsed={time.time()-t0:.0f}s")

    print(f"\n[STATS] {len(event_packs)} events loaded")
    total_fleet_feats = len(event_packs[0]["fleet_cols"]) if event_packs else 0
    print(f"        Fleet-NBM features per event: {total_fleet_feats}")

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

    # Scale-adjusted weighting
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

    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<28} {'n_test':>6} "
          f"{'AUC-ev':>7} {'AUC-evsf':>8} {'AUC-st':>7}")
    print("-" * 85)

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
        y_ev = ep["is_anomaly_event"][test_mask]
        y_st = ep["is_anomaly_status"][test_mask]
        ens_t = ens_s[test_mask]
        status_test = ep["is_anomaly_status"][test_mask]

        # Event-window AUC (all test rows)
        auc_ev = None
        if len(np.unique(y_ev)) == 2:
            auc_ev = float(roc_auc_score(y_ev, ens_t))
        # Event-window AUC, status-filtered (only test rows where status==0)
        auc_evsf = None
        sf = (status_test == 0)
        if sf.sum() > 0 and len(np.unique(y_ev[sf])) == 2:
            auc_evsf = float(roc_auc_score(y_ev[sf], ens_t[sf]))
        auc_st = None
        if len(np.unique(y_st)) == 2:
            auc_st = float(roc_auc_score(y_st, ens_t))

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  —   "
        print(f"{ep['event_id']:>4} {ep['event_label']:>8} "
              f"{ep['event_description'][:26]:<28} {len(y_ev):>6} "
              f"{fmt(auc_ev):>7} {fmt(auc_evsf):>8} {fmt(auc_st):>7}")

        pooled_scores.append(ens_t)
        pooled_labels_event.append(y_ev)
        pooled_labels_status.append(y_st)

        results.append({
            "event_id": ep["event_id"],
            "event_type": ep["event_label"],
            "event_description": ep["event_description"],
            "n_test": ep["n_test"],
            "auc_event": auc_ev,
            "auc_event_statfilt": auc_evsf,
            "auc_status": auc_st,
        })

    print(f"\n{'='*80}")
    print(f"  AGGREGATE — v6 (Fleet-NBM features + ensemble)")
    print(f"{'='*80}")

    for key in ["auc_event", "auc_event_statfilt", "auc_status"]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"  {key:<22} mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}  n={len(vals)}")

    ps = np.concatenate(pooled_scores)
    ple = np.concatenate(pooled_labels_event)
    pls = np.concatenate(pooled_labels_status)
    pooled_ev = float(roc_auc_score(ple, ps)) if len(np.unique(ple)) == 2 else None
    pooled_st = float(roc_auc_score(pls, ps)) if len(np.unique(pls)) == 2 else None
    # Status-filtered pooled
    # NOTE: pls[i] == 0 iff status_type_id == 0 at that test row
    sf_mask_pooled = (pls == 0)
    pooled_evsf = None
    if sf_mask_pooled.sum() > 0 and len(np.unique(ple[sf_mask_pooled])) == 2:
        pooled_evsf = float(roc_auc_score(ple[sf_mask_pooled], ps[sf_mask_pooled]))
    print(f"\n  POOLED AUC (event):             {pooled_ev}")
    print(f"  POOLED AUC (event, status-filt): {pooled_evsf}")
    print(f"  POOLED AUC (status):             {pooled_st}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Ensemble v6 — Fleet-NBM residual features + VAE+LSTM+TF ensemble",
        "n_fleet_nbms": len(nbms),
        "n_features_total": len(shared_cols),
        "n_features_fleet": len([c for c in shared_cols if c.startswith("fleet_nbm_")]),
        "mean_auc_event": float(np.mean([r["auc_event"] for r in results
                                            if r.get("auc_event") is not None])),
        "mean_auc_event_statfilt": float(np.mean([r["auc_event_statfilt"] for r in results
                                                    if r.get("auc_event_statfilt") is not None])),
        "mean_auc_status": float(np.mean([r["auc_status"] for r in results
                                             if r.get("auc_status") is not None])),
        "pooled_auc_event": pooled_ev,
        "pooled_auc_event_statfilt": pooled_evsf,
        "pooled_auc_status": pooled_st,
        "per_event": results,
        "total_time_s": time.time() - t0,
    }
    with (RESULTS_DIR / "care_ensemble_v6.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Saved: {RESULTS_DIR / 'care_ensemble_v6.json'}")
    print(f"[TIME] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
