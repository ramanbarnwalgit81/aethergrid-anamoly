"""Row-level rank-average ensemble of PINN-v2 stacker + THEMIS LOF scores.

Premise
-------
Per-event diagnostics show PINN-v2 and THEMIS are complementary on CARE
Farm A: PINN-v2 dominates generator/bearing-thermal events whose physics
residuals dominate; THEMIS dominates events whose physics residuals are
absent (transformer event 68: PINN-v2 0.04 vs THEMIS 0.62). A row-level
rank-average therefore has a credible chance of beating either alone.

Inputs
------
    docs/results/pinn_stacker_y_{event,precursor,status}.json
        — per_row[event_id] = {stacker_score, v7_score, labels} (test-region only)
    docs/results/themis_chronos_care_per_row.json
        — {event_id: {scores, y_event, y_precursor, y_status, test_mask}} (full event)

Alignment
---------
PINN-v2's test-region uses the LAST n_expected rows of the
train_test == "prediction" subset (EWMA prefix loss in v7). THEMIS
provides scores for the full event with a test_mask. We extract THEMIS
test rows then take the trailing n_pinn rows to align. We verify by
checking labels match.

Output
------
    docs/results/pinn_themis_ensemble_care.json
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS = Path("docs/results")


def _rank_pct(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True, method="average").to_numpy()


def _safe_auc(y, s):
    if len(np.unique(y)) != 2:
        return None
    return float(roc_auc_score(y, s))


def main():
    themis_path = RESULTS / "themis_chronos_care_per_row.json"
    if not themis_path.exists():
        raise SystemExit(f"missing {themis_path} — re-run themis_kaggle.py with the per-row patch")
    themis_per_row = json.loads(themis_path.read_text())

    summary = {"per_event": [], "aggregate": {}}

    label_jobs = [
        ("y_event",     "auc_event"),
        ("y_precursor", "auc_precursor"),
        ("y_status",    "auc_status"),
    ]

    aggregate_auc = {}
    pooled_dump = {}

    for label_key, themis_key in label_jobs:
        pinn_path = RESULTS / f"pinn_stacker_{label_key}.json"
        if not pinn_path.exists():
            print(f"[skip] {pinn_path} missing")
            continue
        pinn = json.loads(pinn_path.read_text())
        pinn_rows = pinn.get("per_row", {})
        if not pinn_rows:
            print(f"[skip] {pinn_path} has no per_row block — re-run pinn_stacker.py with the patch")
            continue

        per_event = []
        ens_scores_pool, ens_labels_pool = [], []

        for eid_s, blk in pinn_rows.items():
            eid = int(eid_s)
            pinn_score = np.asarray(blk["stacker_score"], dtype=np.float64)
            pinn_v7    = np.asarray(blk["v7_score"],      dtype=np.float64)
            pinn_lab   = np.asarray(blk["labels"],        dtype=np.int64)
            n_pinn = len(pinn_score)

            tblk = themis_per_row.get(str(eid)) or themis_per_row.get(eid)
            if tblk is None:
                continue

            t_scores = np.asarray(tblk["scores"], dtype=np.float64)
            t_mask   = np.asarray(tblk["test_mask"], dtype=bool)
            t_lab    = np.asarray(tblk[label_key], dtype=np.int64)

            t_scores_test = t_scores[t_mask]
            t_lab_test    = t_lab[t_mask]
            if len(t_scores_test) < n_pinn:
                continue
            t_scores_test = t_scores_test[-n_pinn:]
            t_lab_test    = t_lab_test[-n_pinn:]

            label_match = float((t_lab_test == pinn_lab).mean())

            r_pinn   = _rank_pct(pinn_score)
            r_themis = _rank_pct(t_scores_test)
            ens      = 0.5 * r_pinn + 0.5 * r_themis

            auc_pinn   = _safe_auc(pinn_lab, pinn_score)
            auc_themis = _safe_auc(pinn_lab, t_scores_test)
            auc_ens    = _safe_auc(pinn_lab, ens)
            auc_v7     = _safe_auc(pinn_lab, pinn_v7)

            per_event.append({
                "event_id":         eid,
                "n_rows":           int(n_pinn),
                "label_match_frac": round(label_match, 4),
                "auc_v7":           None if auc_v7    is None else round(auc_v7, 4),
                "auc_pinn_v2":      None if auc_pinn  is None else round(auc_pinn, 4),
                "auc_themis":       None if auc_themis is None else round(auc_themis, 4),
                "auc_ensemble":     None if auc_ens   is None else round(auc_ens, 4),
            })
            if auc_ens is not None:
                ens_scores_pool.append(_rank_pct(ens))
                ens_labels_pool.append(pinn_lab)

        def _mean(key):
            vs = [r[key] for r in per_event if r.get(key) is not None]
            return round(float(np.mean(vs)), 4) if vs else None

        agg = {
            "label":             label_key,
            "n_events":          len(per_event),
            "mean_auc_v7":       _mean("auc_v7"),
            "mean_auc_pinn_v2":  _mean("auc_pinn_v2"),
            "mean_auc_themis":   _mean("auc_themis"),
            "mean_auc_ensemble": _mean("auc_ensemble"),
        }
        if ens_scores_pool:
            ps = np.concatenate(ens_scores_pool)
            pl = np.concatenate(ens_labels_pool)
            if len(np.unique(pl)) == 2:
                agg["pooled_auc_ensemble"] = round(float(roc_auc_score(pl, ps)), 4)

        aggregate_auc[label_key] = agg
        summary["per_event"].append({"label": label_key, "rows": per_event})

        print(f"\n=== {label_key} ===")
        print(f"{'ID':>4} {'n':>5} {'match':>6} {'v7':>7} {'PINN':>7} {'THEM':>7} {'ENS':>7}")
        print("-" * 50)
        for r in per_event:
            def f(k): return f"{r[k]:.3f}" if r.get(k) is not None else "  —  "
            print(f"{r['event_id']:>4} {r['n_rows']:>5} {r['label_match_frac']:>6.3f} "
                  f"{f('auc_v7'):>7} {f('auc_pinn_v2'):>7} {f('auc_themis'):>7} {f('auc_ensemble'):>7}")
        print(f"  Mean v7       = {agg['mean_auc_v7']}")
        print(f"  Mean PINN-v2  = {agg['mean_auc_pinn_v2']}")
        print(f"  Mean THEMIS   = {agg['mean_auc_themis']}")
        print(f"  Mean ENSEMBLE = {agg['mean_auc_ensemble']}   "
              f"pooled = {agg.get('pooled_auc_ensemble')}")

    summary["aggregate"] = aggregate_auc
    out = RESULTS / "pinn_themis_ensemble_care.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[OK] saved {out}")


if __name__ == "__main__":
    main()
