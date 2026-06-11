"""Cross-farm v7 per-event AUC bootstrap on all 44 CARE anomaly events.

The per-event score arrays were serialised only for Farm A. Farms B and C
only have per-event AUC summaries in the ensemble-replication JSONs. We
therefore cannot pool the PINN-v2 STACKER uplift across farms (which
requires per-row scores to re-fit), but we CAN bootstrap the v7 base
per-event AUC distribution on all 44 anomaly events under the
status_type_id label (the one Farm B/C results use).

This gives the reader:
  - a pooled v7 base AUC with 95% CI on n=44
  - per-farm decomposition
  - direct comparison point for the Farm-A PINN-v2 stacker result

Output:  docs/results/crossfarm_v7_bootstrap.json
Usage:   python -m src.benchmark.crossfarm_v7_bootstrap
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np

RESULTS_DIR = Path("docs/results")
RNG = np.random.default_rng(42)
N_BOOT = 2000


def farm_aucs(farm_file: str, auc_key: str) -> list:
    d = json.load((RESULTS_DIR / farm_file).open())
    return [r[auc_key] for r in d.get("per_event", [])
             if r.get("event_type") == "anomaly" and auc_key in r]


def bootstrap_mean(arr: np.ndarray):
    n = len(arr)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        means[b] = arr[RNG.integers(0, n, n)].mean()
    return {
        "n": int(n),
        "observed_mean": round(float(arr.mean()), 4),
        "bootstrap_95ci": [round(float(np.quantile(means, 0.025)), 4),
                             round(float(np.quantile(means, 0.975)), 4)],
    }


def main():
    # Farm A: from care_ensemble_v7.json we have summary only. Better: use the
    # per-event-scores JSON and compute per-event AUC ourselves for event-window.
    with (RESULTS_DIR / "care_ensemble_v7_per_event_scores.json").open() as f:
        v7a = json.load(f)
    from sklearn.metrics import roc_auc_score
    a_event, a_status = [], []
    for eid, rec in v7a.items():
        s = np.asarray(rec["scores"], dtype=float)
        ye = np.asarray(rec["y_event"], dtype=int)
        ys = np.asarray(rec["y_status"], dtype=int)
        if len(np.unique(ye)) == 2:
            a_event.append(roc_auc_score(ye, s))
        if len(np.unique(ys)) == 2:
            a_status.append(roc_auc_score(ys, s))

    b_event = farm_aucs("care_farm_b_ensemble.json", "auc_event")
    b_status = farm_aucs("care_farm_b_ensemble.json", "auc_status")
    c_event = farm_aucs("care_farm_c_ensemble.json", "auc_event")
    c_status = farm_aucs("care_farm_c_ensemble.json", "auc_status")

    pooled_event = np.array(a_event + b_event + c_event, dtype=float)
    pooled_status = np.array(a_status + b_status + c_status, dtype=float)

    out = {
        "label_event_window": {
            "per_farm": {
                "A": bootstrap_mean(np.asarray(a_event, dtype=float)),
                "B": bootstrap_mean(np.asarray(b_event, dtype=float)),
                "C": bootstrap_mean(np.asarray(c_event, dtype=float)),
            },
            "pooled": bootstrap_mean(pooled_event),
        },
        "label_status": {
            "per_farm": {
                "A": bootstrap_mean(np.asarray(a_status, dtype=float)),
                "B": bootstrap_mean(np.asarray(b_status, dtype=float)),
                "C": bootstrap_mean(np.asarray(c_status, dtype=float)),
            },
            "pooled": bootstrap_mean(pooled_status),
        },
    }
    (RESULTS_DIR / "crossfarm_v7_bootstrap.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
