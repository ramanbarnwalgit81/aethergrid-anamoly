"""
Ensemble SOTA v7 — The money shot.

Same model as v6 (Fleet-NBM features + VAE+LSTM+TF ensemble) but evaluated
against THREE label definitions side by side:
  1. event_window     — `[event_start_id, event_end_id]` (retrospective)
  2. event_statfilt   — event_window restricted to status_type_id == 0 rows
  3. precursor_window — `[precursor_start_id, event_end_id]` (genuinely predictive)

The hypothesis tested: if v6 already matches SOTA on status-code anomalies
(B.3) but under-performs on event-window (B.4 = 0.598 pooled AUC), then
the model may already be DETECTING the precursor drift — just not getting
credit for it because the retrospective label window starts too late.
Evaluating against CARE-Precursor labels measures what the model actually
accomplishes.

Expected behavior:
  - If hypothesis holds: precursor AUC >> event AUC (maybe 0.75+)
  - If hypothesis fails: all three labels cluster at 0.55-0.65

Either outcome is publishable.

Usage:
    python -m src.benchmark.ensemble_sota_v7
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

from src.benchmark.features import engineered_feature_columns
from src.benchmark.vae_baseline import train_vae, vae_anomaly_score
from src.benchmark.lstm_ae import train_lstm_ae, lstm_ae_anomaly_score
from src.benchmark.transformer_ae import train_transformer_ae, transformer_ae_anomaly_score
from src.benchmark.care_sota import build_event_features, ewma_smooth
from src.benchmark.fleet_nbm import (
    load_nbm, TARGET_COLS as NBM_TARGETS,
)
from src.benchmark.ensemble_sota_v6 import (
    compute_fleet_residual_features, build_pool, prepare_event, z_normalize,
)


CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")
DATASETS_DIR = CARE_DIR / "datasets"
EVENT_INFO = CARE_DIR / "event_info.csv"
PRECURSOR_DIR = Path("data/benchmark/care_precursor")
RESULTS_DIR = Path("docs/results")

SEQ_LEN = 12
VAE_HIDDEN = 128
VAE_LATENT = 16
LSTM_HIDDEN = 64
TRANSFORMER_DMODEL = 64
TRANSFORMER_HEADS = 4
EWMA_ALPHA = 0.3


def load_event_v7(event_row, precursor_row, nbms, use_fft=True):
    event_id = int(event_row["event_id"])
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return None

    df_raw = pd.read_csv(csv_path, sep=";", low_memory=False)
    train_test_raw = df_raw["train_test"].values if "train_test" in df_raw.columns else None
    status_type_raw = df_raw["status_type_id"].values if "status_type_id" in df_raw.columns else None

    enriched = build_event_features(df_raw, use_fft=use_fft)
    fleet_features = compute_fleet_residual_features(enriched, nbms)
    for col, vals in fleet_features.items():
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

    # Three label variants
    is_anomaly_event = np.zeros(n, dtype=int)
    is_anomaly_precursor = np.zeros(n, dtype=int)
    is_anomaly_status = (status_arr != 0).astype(int)

    if event_row["event_label"] == "anomaly":
        s = int(event_row["event_start_id"])
        e = int(event_row["event_end_id"])
        is_anomaly_event[s:e + 1] = 1

        # Precursor: use precursor_start_id if found, else fall back to event_start
        if precursor_row is not None:
            ps = int(precursor_row.get("precursor_start_id", s))
            is_anomaly_precursor[ps:e + 1] = 1
        else:
            is_anomaly_precursor[s:e + 1] = 1

    return {
        "event_id": event_id,
        "event_label": event_row["event_label"],
        "event_description": str(event_row.get("event_description", ""))[:40],
        "enriched": enriched,
        "feature_cols": feature_cols,
        "fleet_cols": fleet_cols,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "is_anomaly_event": is_anomaly_event,
        "is_anomaly_precursor": is_anomaly_precursor,
        "is_anomaly_status": is_anomaly_status,
        "precursor_found": bool(precursor_row.get("precursor_found", False))
                            if precursor_row is not None else False,
        "precursor_lead_hours": float(precursor_row.get("precursor_lead_time_hours", 0.0))
                                  if precursor_row is not None else 0.0,
        "n_rows": n,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def main():
    print("=" * 80)
    print("  ENSEMBLE SOTA v7 — DUAL-LABEL EVAL (event + precursor + statfiltered)")
    print("=" * 80)

    # Load Fleet-NBMs
    nbms = {}
    for target in NBM_TARGETS:
        b = load_nbm("penmanshiel", target)
        if b is not None:
            nbms[target] = b
    print(f"[NBM] Loaded {len(nbms)} Fleet-NBMs")

    # Load event info + precursor labels
    events = pd.read_csv(EVENT_INFO, sep=";")
    precursor_path = PRECURSOR_DIR / "event_info_precursor_farm_A.csv"
    if precursor_path.exists():
        precursor = pd.read_csv(precursor_path, sep=";")
        precursor_map = {int(r["event_id"]): r.to_dict() for _, r in precursor.iterrows()}
        print(f"[PRECURSOR] Loaded labels for {len(precursor)} events")
    else:
        precursor_map = {}
        print(f"[PRECURSOR] No labels found — run care_precursor.py first")
        return

    t0 = time.time()

    print(f"\n[LOAD] Building rich+fleet features for {len(events)} events...")
    event_packs = []
    for i, (_, row) in enumerate(events.iterrows()):
        pre_row = precursor_map.get(int(row["event_id"]))
        ep = load_event_v7(row, pre_row, nbms)
        if ep is not None:
            event_packs.append(ep)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(events)}] id={row['event_id']} "
                  f"({row['event_label']}) elapsed={time.time()-t0:.0f}s")

    print(f"\n[STATS] {len(event_packs)} events loaded")

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

    # Save models to disk so we don't have to retrain ever again
    MODELS_DIR = Path("models/ensemble_v7")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vae_state": vae_bundle["model"].state_dict(),
        "vae_val": vae_bundle["best_val"],
        "vae_n_features": vae_bundle["n_features"],
        "vae_hidden": VAE_HIDDEN, "vae_latent": VAE_LATENT,
        "lstm_state": lstm_bundle["model"].state_dict() if lstm_bundle.get("model") is not None else None,
        "lstm_val": lstm_bundle["best_val"],
        "lstm_seq_len": SEQ_LEN, "lstm_hidden": LSTM_HIDDEN,
        "lstm_n_features": lstm_bundle["n_features"],
        "tf_state": tf_bundle["model"].state_dict() if tf_bundle.get("model") is not None else None,
        "tf_val": tf_bundle["best_val"],
        "tf_seq_len": SEQ_LEN, "tf_d_model": TRANSFORMER_DMODEL,
        "tf_n_heads": TRANSFORMER_HEADS,
        "tf_n_features": tf_bundle["n_features"],
        "shared_cols": shared_cols,
        "col_medians": col_medians.tolist(),
        "x_center_": scaler.center_.tolist(),
        "x_scale_": scaler.scale_.tolist(),
    }, MODELS_DIR / "ensemble_v7.pt")
    print(f"[SAVE] Ensemble weights saved to {MODELS_DIR}/ensemble_v7.pt")

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
    # Pooled scores + labels for all three definitions
    pooled_scores = []
    pooled_ev = []
    pooled_pre = []
    pooled_st = []
    pooled_sf_mask = []  # True where status_type_id == 0

    print(f"\n{'ID':>4} {'Label':>8} {'Fault':<26} "
          f"{'AUC-ev':>7} {'AUC-sf':>7} {'AUC-pre':>8} "
          f"{'lead_hr':>7}")
    print("-" * 90)

    per_event_scores = {}  # event_id -> {"scores": arr, "labels_...": arr}

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
        y_pre = ep["is_anomaly_precursor"][test_mask]
        y_st = ep["is_anomaly_status"][test_mask]
        ens_t = ens_s[test_mask]
        status_test = ep["is_anomaly_status"][test_mask]
        sf_mask_test = (status_test == 0)

        auc_ev = float(roc_auc_score(y_ev, ens_t)) if len(np.unique(y_ev)) == 2 else None
        auc_pre = float(roc_auc_score(y_pre, ens_t)) if len(np.unique(y_pre)) == 2 else None
        auc_sf = None
        if sf_mask_test.sum() > 0 and len(np.unique(y_ev[sf_mask_test])) == 2:
            auc_sf = float(roc_auc_score(y_ev[sf_mask_test], ens_t[sf_mask_test]))
        auc_st = float(roc_auc_score(y_st, ens_t)) if len(np.unique(y_st)) == 2 else None

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  —   "
        print(f"{ep['event_id']:>4} {ep['event_label']:>8} "
              f"{ep['event_description'][:24]:<26} "
              f"{fmt(auc_ev):>7} {fmt(auc_sf):>7} {fmt(auc_pre):>8} "
              f"{ep['precursor_lead_hours']:>7.1f}")

        # Store per-event scores for downstream analyses
        per_event_scores[ep["event_id"]] = {
            "scores": ens_t.tolist(),
            "y_event": y_ev.tolist(),
            "y_precursor": y_pre.tolist(),
            "y_status": y_st.tolist(),
            "sf_mask": sf_mask_test.tolist(),
        }

        pooled_scores.append(ens_t)
        pooled_ev.append(y_ev)
        pooled_pre.append(y_pre)
        pooled_st.append(y_st)
        pooled_sf_mask.append(sf_mask_test)

        results.append({
            "event_id": ep["event_id"],
            "event_type": ep["event_label"],
            "event_description": ep["event_description"],
            "n_test": ep["n_test"],
            "auc_event": auc_ev,
            "auc_event_statfilt": auc_sf,
            "auc_precursor": auc_pre,
            "auc_status": auc_st,
            "precursor_found": ep["precursor_found"],
            "precursor_lead_hours": ep["precursor_lead_hours"],
        })

    print(f"\n{'='*90}")
    print(f"  AGGREGATE — v7 (dual-label evaluation)")
    print(f"{'='*90}")

    for key in ["auc_event", "auc_event_statfilt", "auc_precursor", "auc_status"]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"  {key:<22} mean={np.mean(vals):.4f}  "
                  f"median={np.median(vals):.4f}  n={len(vals)}")

    ps = np.concatenate(pooled_scores)
    ple = np.concatenate(pooled_ev)
    plp = np.concatenate(pooled_pre)
    pls = np.concatenate(pooled_st)
    psf = np.concatenate(pooled_sf_mask)

    pooled_ev_auc = float(roc_auc_score(ple, ps)) if len(np.unique(ple)) == 2 else None
    pooled_pre_auc = float(roc_auc_score(plp, ps)) if len(np.unique(plp)) == 2 else None
    pooled_st_auc = float(roc_auc_score(pls, ps)) if len(np.unique(pls)) == 2 else None
    pooled_evsf = None
    if psf.sum() > 0 and len(np.unique(ple[psf])) == 2:
        pooled_evsf = float(roc_auc_score(ple[psf], ps[psf]))
    pooled_presf = None
    if psf.sum() > 0 and len(np.unique(plp[psf])) == 2:
        pooled_presf = float(roc_auc_score(plp[psf], ps[psf]))

    print(f"\n  POOLED AUC:")
    print(f"    event:                   {pooled_ev_auc}")
    print(f"    event (status-filtered): {pooled_evsf}")
    print(f"    precursor:               {pooled_pre_auc}")
    print(f"    precursor (status-filt): {pooled_presf}")
    print(f"    status:                  {pooled_st_auc}")

    # Sub-analysis: events WITH precursor vs events WITHOUT
    with_pre = [r for r in results if r.get("precursor_found") and r.get("auc_precursor") is not None]
    wo_pre = [r for r in results if (not r.get("precursor_found")) and r["event_type"] == "anomaly"
               and r.get("auc_event") is not None]
    if with_pre:
        print(f"\n  Events WITH sustained precursor (n={len(with_pre)}):")
        print(f"    mean AUC (event):     {np.mean([r['auc_event'] for r in with_pre]):.4f}")
        print(f"    mean AUC (precursor): {np.mean([r['auc_precursor'] for r in with_pre]):.4f}")
    if wo_pre:
        print(f"\n  Events WITHOUT sustained precursor (n={len(wo_pre)}):")
        print(f"    mean AUC (event):     {np.mean([r['auc_event'] for r in wo_pre]):.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Ensemble v7 — Fleet-NBM features + dual-label evaluation",
        "n_features_total": len(shared_cols),
        "n_features_fleet": len([c for c in shared_cols if c.startswith("fleet_nbm_")]),
        "ensemble_weights": {"vae": float(w[0]), "lstm": float(w[1]),
                              "transformer": float(w[2])},
        "pooled_auc_event": pooled_ev_auc,
        "pooled_auc_event_statfilt": pooled_evsf,
        "pooled_auc_precursor": pooled_pre_auc,
        "pooled_auc_precursor_statfilt": pooled_presf,
        "pooled_auc_status": pooled_st_auc,
        "per_event": results,
        "total_time_s": time.time() - t0,
    }
    with (RESULTS_DIR / "care_ensemble_v7.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save per-event scores for reuse (calibration, conformal, Streamlit demo)
    with (RESULTS_DIR / "care_ensemble_v7_per_event_scores.json").open("w") as f:
        json.dump(per_event_scores, f, indent=2, default=str)

    print(f"\n[OK] Saved summary: {RESULTS_DIR / 'care_ensemble_v7.json'}")
    print(f"[OK] Saved scores:  {RESULTS_DIR / 'care_ensemble_v7_per_event_scores.json'}")
    print(f"[TIME] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
