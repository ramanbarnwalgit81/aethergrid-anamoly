"""THEMIS frozen-encoder + LOF anomaly detector for CARE Farm A.

Drop into a Kaggle notebook cell after windgpt_kaggle.py has run (the dataset
must be unpacked at /kaggle/working/aethergrid-anomaly).  No fine-tuning,
no training — pure inference + sklearn LocalOutlierFactor on Chronos-Bolt
encoder embeddings.

Reference: THEMIS (arxiv 2510.03911, 2025).  Frozen-encoder + LOF was
reported to beat MOMENT/Chronos forecast-residual scoring on multiple
TSAD benchmarks.  We test whether it also beats our PINN-v2 stacker
(0.577 / 0.557) on CARE Farm A.

Output:  docs/results/themis_chronos_care.json
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

# ─── Paths ─────────────────────────────────────────────────────────────
WORK = Path("/kaggle/working/aethergrid-anomaly")
CARE_DIR = WORK / "data/real_scada/care/extracted/Wind Farm A/Wind Farm A"
RESULTS_DIR = WORK / "docs/results"

# ─── Configuration ─────────────────────────────────────────────────────
MODEL_NAME = "amazon/chronos-bolt-base"
TARGET_SIGNALS = [
    "sensor_12_avg",   # gearbox oil
    "sensor_0_avg",    # ambient
    "power_30_avg",    # active power
    "wind_speed_3_avg",  # wind speed
    "sensor_18_avg",   # rotor / generator RPM
]
CONTEXT_LEN = 256
STRIDE = 32                 # embedding extraction stride (rows)
LOF_N_NEIGHBORS = 30
SUBSAMPLE_FOR_LOF = 5000    # LOF on at most this many rows per event (speed)


def main():
    import torch
    from chronos import ChronosBoltPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[1/4] Loading FROZEN {MODEL_NAME} on {device} ({dtype})...")
    t0 = time.time()
    pipeline = ChronosBoltPipeline.from_pretrained(
        MODEL_NAME, device_map=device, torch_dtype=dtype)
    # Disable gradients globally — frozen encoder
    for p in pipeline.model.parameters():
        p.requires_grad = False
    pipeline.model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"[2/4] Loading CARE Farm A event metadata...")
    events_df = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    anomaly_events = events_df[events_df["event_label"] == "anomaly"]
    pre_path = WORK / "data/benchmark/care_precursor/event_info_precursor_farm_A.csv"
    pre_df = pd.read_csv(pre_path, sep=";") if pre_path.exists() else pd.DataFrame()
    precursor_map = {int(r["event_id"]): r for _, r in pre_df.iterrows()}
    print(f"  {len(anomaly_events)} anomaly events to score")

    print(f"[3/4] Scoring each event (frozen embed -> LOF)...")
    results = []
    per_row_dump = {}
    t_total = time.time()
    for i, (_, row) in enumerate(anomaly_events.iterrows(), start=1):
        event_id = int(row["event_id"])
        csv_path = CARE_DIR / "datasets" / f"{event_id}.csv"
        if not csv_path.exists():
            print(f"  [skip] event {event_id}: csv missing")
            continue
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
        n = len(df)

        # ── Extract embeddings per signal at strided positions ──────────
        sig_embeds = []
        for sig in TARGET_SIGNALS:
            if sig not in df.columns:
                continue
            series = df[sig].ffill().fillna(0).to_numpy(dtype=np.float32)
            if series.std() < 1e-6:
                continue
            positions = list(range(CONTEXT_LEN, n, STRIDE))
            if not positions:
                continue
            # Build batch of normalised contexts
            contexts = []
            for t in positions:
                ctx = series[t - CONTEXT_LEN:t]
                mu, sigma = float(ctx.mean()), float(ctx.std()) + 1e-6
                contexts.append((ctx - mu) / sigma)
            ctx_batch = np.stack(contexts).astype(np.float32)
            # Forward through frozen encoder (chunked to avoid OOM)
            embeds_per_pos = []
            chunk_size = 64
            with torch.no_grad():
                for s in range(0, len(ctx_batch), chunk_size):
                    b = torch.tensor(ctx_batch[s:s + chunk_size],
                                       dtype=dtype, device=device)
                    out = pipeline.embed(b)
                    # ChronosBoltPipeline.embed returns (embeddings, loc_scale)
                    emb = out[0] if isinstance(out, (tuple, list)) else out
                    # Pool over patch dimension -> [chunk, hidden]
                    pooled = emb.mean(dim=1).float().cpu().numpy()
                    embeds_per_pos.append(pooled)
            embeds_per_pos = np.concatenate(embeds_per_pos, axis=0)  # [P, hidden]

            # Map to per-row by nearest strided position
            hidden = embeds_per_pos.shape[1]
            per_row = np.zeros((n, hidden), dtype=np.float32)
            for j, t in enumerate(positions):
                lo = max(0, t - STRIDE // 2)
                hi = min(n, t + STRIDE // 2 + (1 if STRIDE % 2 else 0))
                per_row[lo:hi] = embeds_per_pos[j]
            per_row[:CONTEXT_LEN] = embeds_per_pos[0]    # leading pad
            per_row[positions[-1] + STRIDE // 2:] = embeds_per_pos[-1]   # trailing pad
            sig_embeds.append(per_row)

        if not sig_embeds:
            print(f"  [skip] event {event_id}: no usable signals")
            continue

        full_embed = np.concatenate(sig_embeds, axis=1)
        # Per-feature z-normalize so LOF distances are scale-invariant
        full_embed = (full_embed - full_embed.mean(axis=0)) \
                     / (full_embed.std(axis=0) + 1e-6)

        # ── LOF on a subsampled set, score all rows by predicting on full ──
        # (LOF in novelty=False mode scores the same data it was fit on, so
        # subsampling preserves correctness as long as we score the full set.)
        n_lof = min(SUBSAMPLE_FOR_LOF, n)
        if n_lof < n:
            idx = np.linspace(0, n - 1, n_lof).astype(int)
            X_fit = full_embed[idx]
        else:
            X_fit = full_embed
        n_nbrs = min(LOF_N_NEIGHBORS, max(5, n_lof // 100))
        lof = LocalOutlierFactor(n_neighbors=n_nbrs, novelty=True, n_jobs=-1)
        lof.fit(X_fit)
        # decision_function: higher = more inlier; negate to make higher=more anomalous
        scores = -lof.decision_function(full_embed)

        # ── Labels ─────────────────────────────────────────────────────
        s_idx = int(row.get("event_start_id", -1))
        e_idx = int(row.get("event_end_id", -1))
        y_event = np.zeros(n, dtype=int)
        if s_idx >= 0 and e_idx >= s_idx:
            y_event[s_idx:e_idx + 1] = 1

        y_precursor = y_event.copy()
        pre_row = precursor_map.get(event_id)
        if pre_row is not None:
            ps = int(pre_row.get("precursor_start_id", s_idx))
            if 0 <= ps < s_idx:
                y_precursor[ps:e_idx + 1] = 1

        y_status = np.zeros(n, dtype=int)
        if "status_type_id" in df.columns:
            y_status = (df["status_type_id"].fillna(0).astype(int).to_numpy() != 0).astype(int)

        test_mask = (df.get("train_test") == "prediction").to_numpy() \
            if "train_test" in df.columns else np.ones(n, dtype=bool)

        def _auc(y):
            yt = y[test_mask]; st = scores[test_mask]
            if len(np.unique(yt)) != 2:
                return None
            return float(roc_auc_score(yt, st))

        auc_ev = _auc(y_event)
        auc_pre = _auc(y_precursor)
        auc_st = _auc(y_status)

        results.append({
            "event_id":      event_id,
            "fault":         str(row.get("event_description", ""))[:30],
            "n_rows":        int(n),
            "auc_event":     auc_ev,
            "auc_precursor": auc_pre,
            "auc_status":    auc_st,
        })
        # Persist per-row scores + labels + test_mask for downstream ensembling.
        per_row_dump[int(event_id)] = {
            "scores":      [float(x) for x in scores],
            "y_event":     [int(x) for x in y_event],
            "y_precursor": [int(x) for x in y_precursor],
            "y_status":    [int(x) for x in y_status],
            "test_mask":   [bool(x) for x in test_mask],
        }
        print(f"  [{i:>2}/{len(anomaly_events)}] event {event_id:>3}  "
              f"event={auc_ev}  precursor={auc_pre}  status={auc_st}  "
              f"({(time.time()-t_total)/60:.1f} min elapsed)")

    # ── Aggregate ───────────────────────────────────────────────────────
    def _mean(key):
        v = [r[key] for r in results if r.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    out = {
        "method":             "THEMIS (frozen Chronos-Bolt-base + LOF on row embeddings)",
        "model":              MODEL_NAME,
        "target_signals":     TARGET_SIGNALS,
        "context_len":        CONTEXT_LEN,
        "stride":             STRIDE,
        "lof_n_neighbors":    LOF_N_NEIGHBORS,
        "lof_subsample":      SUBSAMPLE_FOR_LOF,
        "per_event":          results,
        "mean_auc_event":     _mean("auc_event"),
        "mean_auc_precursor": _mean("auc_precursor"),
        "mean_auc_status":    _mean("auc_status"),
        "elapsed_min":        round((time.time() - t_total) / 60, 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "themis_chronos_care.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    per_row_path = RESULTS_DIR / "themis_chronos_care_per_row.json"
    per_row_path.write_text(json.dumps(per_row_dump, default=str))
    print(f"[OK] Saved per-row scores: {per_row_path}")

    print(f"\n[4/4] {'='*60}")
    print(f"  THEMIS frozen-encoder + LOF on CARE Farm A")
    print(f"{'='*60}")
    print(f"  Mean AUC event:     {out['mean_auc_event']}")
    print(f"  Mean AUC precursor: {out['mean_auc_precursor']}")
    print(f"  Mean AUC status:    {out['mean_auc_status']}")
    print(f"  Elapsed:            {out['elapsed_min']:.1f} min")
    print(f"  Reference: PINN-v2 stacker = 0.577 / 0.557 / 0.646")
    print(f"             Chronos-T5 zero-shot = 0.465 / 0.476 / —")
    print(f"             WindGPT LoRA = 0.472 / 0.450 / 0.465")
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
