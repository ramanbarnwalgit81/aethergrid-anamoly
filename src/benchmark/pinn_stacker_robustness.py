"""Small-sample robustness battery for the PINN-v2 per-event AUC uplift.

The headline uplift on CARE Farm A is computed over a small number of events
(n=12 event-window, n=10 CARE-Precursor), and one transformer-fault event
(id 68) on which the reconstruction stacker regresses pulls the mean down. A
mean-based bootstrap is therefore vulnerable to a "one event drives it"
objection. This module re-tests the same per-event deltas with statistics that
do not depend on the magnitude of any single point:

  * median uplift (vs the mean), to show central tendency is not outlier-driven,
  * exact Wilcoxon signed-rank test (signed ranks, not magnitudes),
  * exact sign-flip permutation test (no distributional assumption),
  * sign test (binomial on the count of positive events; the weakest test,
    reported for completeness even when it does not clear 0.05),
  * leave-one-event-out: the Wilcoxon p-value and mean with each single event
    removed, which exposes whether any one event drives the result.

Deterministic: every test here is exact (no RNG). Run from the repo root:
    python -m src.benchmark.pinn_stacker_robustness
Writes docs/results/pinn_stacker_robustness.json.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("docs/results")

TASKS = {
    "event_window": "pinn_stacker_y_event.json",
    "precursor": "pinn_stacker_y_precursor.json",
}


def _deltas(fname: str):
    d = json.loads((RESULTS_DIR / fname).read_text())
    per = d["per_event"]
    ids = [int(r["event_id"]) for r in per]
    delta = np.array([float(r["pinn_stacker_auc"]) - float(r["base_v7_auc"]) for r in per])
    return ids, delta


def _exact_signflip_p(delta: np.ndarray) -> float:
    """One-sided exact paired permutation p-value under sign flips of the deltas."""
    obs = float(delta.mean())
    n = len(delta)
    ge = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        if (delta * np.asarray(signs)).mean() >= obs - 1e-12:
            ge += 1
    return ge / total


def _battery(ids, delta):
    n = len(delta)
    n_pos = int((delta > 0).sum())
    wil = stats.wilcoxon(delta, alternative="greater", mode="exact")
    sign = stats.binomtest(n_pos, n, 0.5, alternative="greater")
    loo = []
    for i in range(n):
        dd = np.delete(delta, i)
        w = stats.wilcoxon(dd, alternative="greater", mode="exact")
        loo.append({
            "dropped_event": ids[i],
            "mean": round(float(dd.mean()), 4),
            "median": round(float(np.median(dd)), 4),
            "wilcoxon_p": round(float(w.pvalue), 4),
        })
    # The single most influential (downward) event: the one whose removal most
    # raises the mean.
    drag = max(loo, key=lambda r: r["mean"])
    loo_max_p = max(r["wilcoxon_p"] for r in loo)
    return {
        "n": n,
        "n_positive": n_pos,
        "mean_uplift": round(float(delta.mean()), 4),
        "median_uplift": round(float(np.median(delta)), 4),
        "wilcoxon_signed_rank_p_onesided": round(float(wil.pvalue), 4),
        "sign_test_p_onesided": round(float(sign.pvalue), 4),
        "signflip_permutation_p_onesided": round(_exact_signflip_p(delta), 4),
        "leave_one_out": loo,
        "leave_one_out_worst_wilcoxon_p": round(loo_max_p, 4),
        "most_influential_event": drag["dropped_event"],
        "mean_without_most_influential": drag["mean"],
        "wilcoxon_p_without_most_influential": drag["wilcoxon_p"],
        "per_event_delta": {str(i): round(float(x), 4) for i, x in zip(ids, delta)},
    }


def main():
    out = {
        "method": "Small-sample robustness battery for the PINN-v2 per-event AUC uplift "
                  "(exact Wilcoxon, exact sign-flip permutation, sign test, leave-one-event-out).",
        "tasks": {},
    }
    for task, fname in TASKS.items():
        ids, delta = _deltas(fname)
        out["tasks"][task] = _battery(ids, delta)
    (RESULTS_DIR / "pinn_stacker_robustness.json").write_text(json.dumps(out, indent=2))
    print("[OK] docs/results/pinn_stacker_robustness.json")
    for task, r in out["tasks"].items():
        print(f"\n{task}: n={r['n']}  {r['n_positive']}/{r['n']} positive  "
              f"mean={r['mean_uplift']:+.3f}  median={r['median_uplift']:+.3f}")
        print(f"  Wilcoxon p={r['wilcoxon_signed_rank_p_onesided']}  "
              f"permutation p={r['signflip_permutation_p_onesided']}  "
              f"sign-test p={r['sign_test_p_onesided']}")
        print(f"  leave-one-out worst Wilcoxon p={r['leave_one_out_worst_wilcoxon_p']}; "
              f"dropping event {r['most_influential_event']} -> mean {r['mean_without_most_influential']:+.3f}, "
              f"p={r['wilcoxon_p_without_most_influential']}")


if __name__ == "__main__":
    main()
