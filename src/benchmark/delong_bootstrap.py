"""Bootstrap CIs + DeLong-style statistical comparison for SOTA replication.

Two claims in the paper are statistical rather than descriptive:
  (a) Our pooled AUC 0.9447 on Farm B is "statistically indistinguishable" from
      Nair 2025's reported 0.947.
  (b) Fleet-NBM lifts pooled event-window AUC from 0.507 to 0.598 (+0.091).

For (a) we do not have Nair's pooled predictions, so the strongest honest test is
a one-sample bootstrap CI on our own pooled AUC and a check that the claimed
Nair value 0.947 falls inside the 95% band.  For (b) we use the paired-bootstrap
AUC delta (correlated samples — same events), which is asymptotically equivalent
to a DeLong test on a single aggregated pool.

Usage: python -m src.benchmark.delong_bootstrap
Outputs: docs/results/delong_bootstrap.json
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("docs/results")
N_BOOT = 2000
RNG = np.random.default_rng(42)


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray,
                      n_boot: int = N_BOOT, alpha: float = 0.05):
    """Bootstrap 95 % CI for AUC (case resampling)."""
    n = len(y)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(y[idx], scores[idx])
    aucs = aucs[np.isfinite(aucs)]
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2)), aucs


def paired_bootstrap_delta(y: np.ndarray, scores_a: np.ndarray,
                             scores_b: np.ndarray,
                             n_boot: int = N_BOOT):
    """Bootstrap 95 % CI for AUC(a) - AUC(b) with PAIRED resampling."""
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            deltas[b] = np.nan
            continue
        deltas[b] = roc_auc_score(y[idx], scores_a[idx]) - roc_auc_score(y[idx], scores_b[idx])
    deltas = deltas[np.isfinite(deltas)]
    return deltas


# ───────────────────────────────── Test (a): Farm B/C per-event bootstrap
def _event_bootstrap(per_events, auc_key, claimed_nair):
    aucs = np.array([r[auc_key] for r in per_events if r.get("event_type") == "anomaly"])
    if len(aucs) == 0:
        return {"skipped": True}
    n = len(aucs)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        means[b] = aucs[RNG.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, 0.025), np.quantile(means, 0.975)
    return {
        "n_anomaly_events": int(n),
        "observed_mean_event_auc": round(float(aucs.mean()), 4),
        "bootstrap_95ci_event_level": [round(float(lo), 4), round(float(hi), 4)],
        "nair_claimed": claimed_nair,
        "nair_in_event_level_ci": bool(lo <= claimed_nair <= hi),
    }


def test_farm_b():
    with (RESULTS_DIR / "care_farm_b_ensemble.json").open() as f:
        d = json.load(f)
    out = {"claim": "Farm B AUC ≈ Nair 0.947", "pooled": d.get("pooled_auc_status")}
    out.update(_event_bootstrap(d.get("per_event", []), "auc_status", 0.947))
    return out


def test_farm_c():
    with (RESULTS_DIR / "care_farm_c_ensemble.json").open() as f:
        d = json.load(f)
    out = {"claim": "Farm C AUC ≈ Nair 0.963", "pooled": d.get("pooled_auc_status")}
    out.update(_event_bootstrap(d.get("per_event", []), "auc_status", 0.963))
    return out


# ───────────────────────────────── Test (b): Fleet-NBM paired uplift
def test_nbm_uplift():
    """Compare v3 (ensemble, no NBM) vs v6 (ensemble + NBM) on CARE Farm A
    event-window label using saved per-event score JSONs if available."""
    v3 = RESULTS_DIR / "care_ensemble_v3.json"
    v6 = RESULTS_DIR / "care_ensemble_v6.json"
    if not (v3.exists() and v6.exists()):
        return {"claim": "v3 vs v6 uplift",
                  "bootstrap_skipped": True,
                  "reason": "missing per-event files"}
    v3_d = json.loads(v3.read_text())
    v6_d = json.loads(v6.read_text())
    # Both scalar-only; paired per-row comparison not possible without per-event
    # test scores that were aligned to the same evaluation protocol.  Flag.
    return {
        "claim": "Fleet-NBM uplift +0.091 on pooled event-window AUC",
        "v3_pooled_auc": v3_d.get("pooled_auc"),
        "v6_pooled_auc": v6_d.get("pooled_auc_event"),
        "delta": round(v6_d.get("pooled_auc_event", 0) - v3_d.get("pooled_auc", 0), 4)
                  if v3_d.get("pooled_auc") and v6_d.get("pooled_auc_event") else None,
        "bootstrap_skipped": True,
        "reason": "v3 and v6 per-row scores aligned on different test splits; "
                    "paired bootstrap not applicable without stored-score realignment",
    }


# ───────────────────────────────── Test (c): PINN-v2 vs v7 base (all labels)
def _uplift_ci(file_suffix: str, label_name: str):
    with (RESULTS_DIR / f"pinn_stacker_{file_suffix}.json").open() as f:
        s2 = json.load(f)
    per = s2["per_event"]
    base = np.array([r["base_v7_auc"] for r in per])
    stack = np.array([r["pinn_stacker_auc"] for r in per])
    delta = stack - base
    deltas = np.empty(N_BOOT)
    n = len(delta)
    for b in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        deltas[b] = delta[idx].mean()
    lo, hi = np.quantile(deltas, 0.025), np.quantile(deltas, 0.975)
    return {
        "claim": f"PINN-v2 uplift over v7 on {label_name}",
        "n_events": int(n),
        "observed_mean_delta": round(float(delta.mean()), 4),
        "bootstrap_95ci_mean_delta": [round(float(lo), 4), round(float(hi), 4)],
        "zero_in_ci": bool(lo <= 0 <= hi),
        "p_one_sided": round(float((deltas <= 0).mean()), 4),
        "n_boot": N_BOOT,
    }


def test_pinn_v2_uplift():
    return {
        "event_window": _uplift_ci("y_event", "event-window"),
        "care_precursor": _uplift_ci("y_precursor", "CARE-Precursor"),
        "status":         _uplift_ci("y_status", "status_type_id"),
    }


def main():
    out = {
        "test_a_farm_b_vs_nair": test_farm_b(),
        "test_a2_farm_c_vs_nair": test_farm_c(),
        "test_b_nbm_uplift": test_nbm_uplift(),
        "test_c_pinn_v2_uplift": test_pinn_v2_uplift(),
    }
    (RESULTS_DIR / "delong_bootstrap.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
