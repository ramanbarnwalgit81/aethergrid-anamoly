"""
Causal Conformal Risk Control (CCRC) — novel theoretical framework.

FIRST combination of PCMCI causal discovery + Conformal Risk Control in
any application domain, to our knowledge.

CONCEPT
-------
Standard CRC (Angelopoulos et al. 2022) gives distribution-free risk bounds
under exchangeability. When exchangeability fails across groups (e.g., fault
types on CARE), marginal bounds VIOLATE per-group (§B.12 showed Generator
bearing FPR = 0.63).

CCRC adds a CAUSAL stratification:
  1. Use PCMCI+ to learn the causal DAG over signals (from §B.15)
  2. Cluster events by their causal-DAG signature (similar parent structure)
  3. Apply CRC PER causal cluster
  4. Output: risk bounds that are GROUP-CONDITIONAL along the learned
     causal signature, not merely along observed covariates (fault type)

Why novel
---------
- Group-conditional CP (Bostrom et al. 2017, Cauchois et al. 2021) requires
  groups to be KNOWN. CCRC discovers groups from the causal DAG.
- Cauchois et al. 2024 "Robust Predictive Inference" handles distribution
  shift but not via causal structure.
- No published paper combines PCMCI (causal discovery) with CRC (risk
  control) for prediction sets.

Expected finding: CCRC recovers partial coverage on event-window labels
where marginal CRC violated — by using the causal DAG to decide which
calibration samples are exchangeable with the held-out event.

Usage:
    python -m src.benchmark.causal_conformal
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering

RESULTS_DIR = Path("docs/results")
CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")


def load_v7_scores() -> dict:
    with (RESULTS_DIR / "care_ensemble_v7_per_event_scores.json").open() as f:
        return json.load(f)


def load_causal_discovery() -> dict:
    """Load per-event PCMCI+ results from §B.15."""
    path = RESULTS_DIR / "causal_discovery.json"
    with path.open() as f:
        return json.load(f)


def causal_signature(pcmci_parents: dict) -> np.ndarray:
    """
    Convert per-event causal DAG into a fixed-dim signature vector.

    Signature = flattened adjacency-lag matrix (rows = targets, cols = sources,
    value = max absolute coefficient across lags). Ordering follows the 5
    target signals from §B.15.
    """
    targets = [
        "sensor_12_avg",
        "sensor_0_avg",
        "power_30_avg",
        "wind_speed_3_avg",
        "sensor_18_avg",
    ]
    sig = np.zeros((len(targets), len(targets)), dtype=np.float32)
    for i, tgt in enumerate(targets):
        parents = pcmci_parents.get(tgt, [])
        for p in parents:
            src = p.get("source")
            if src in targets:
                j = targets.index(src)
                coef = abs(float(p.get("coefficient", 0.0)))
                sig[i, j] = max(sig[i, j], coef)
    return sig.ravel()


def empirical_fpr(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    n_neg = (labels == 0).sum()
    if n_neg == 0:
        return 0.0
    alerts = (scores >= tau).astype(int)
    return float(((alerts == 1) & (labels == 0)).sum() / n_neg)


def marginal_crc_threshold(calib_scores, calib_labels, target_fpr, n_cand=200):
    candidates = np.quantile(calib_scores, np.linspace(0.001, 0.999, n_cand))
    adj = max(0.0, target_fpr - (1.0 - target_fpr) / max(len(calib_scores), 1))
    best = float(candidates.max())
    for tau in reversed(candidates):
        if empirical_fpr(calib_scores, calib_labels, tau) <= adj:
            best = float(tau)
        else:
            break
    return best


def main():
    print("=" * 80)
    print("  CAUSAL CONFORMAL RISK CONTROL (CCRC) — CARE Farm A")
    print("  First combination of PCMCI causal discovery + CRC")
    print("=" * 80)

    per_event = load_v7_scores()
    causal = load_causal_discovery()["per_event"]

    # Build per-event causal signature + score/label arrays
    signatures = {}
    for rec in causal:
        eid = rec["event_id"]
        parents = rec.get("pcmci_parents", {})
        signatures[int(eid)] = causal_signature(parents)

    all_event_ids = [int(k) for k in per_event.keys()]
    print(f"\n[DATA] {len(all_event_ids)} events with v7 scores")
    print(f"       {len(signatures)} events with causal signatures")

    # Cluster events by causal signature (agglomerative clustering)
    sig_matrix = np.stack([signatures[eid] for eid in all_event_ids
                              if eid in signatures], axis=0)
    clusterable_ids = [eid for eid in all_event_ids if eid in signatures]

    print(f"\n[CLUSTER] Signature matrix shape: {sig_matrix.shape}")
    # Agglomerative with 3 clusters (intuition: bearing/hydraulic/transformer families)
    clusterer = AgglomerativeClustering(
        n_clusters=3, metric="cosine", linkage="average",
    )
    cluster_labels = clusterer.fit_predict(sig_matrix)
    cluster_map = dict(zip(clusterable_ids, cluster_labels.tolist()))
    print(f"  Cluster assignments:")
    for eid, c in sorted(cluster_map.items()):
        print(f"    event {eid:>3} -> cluster {c}")

    target_fpr = 0.05
    results = {
        "marginal_crc": {},
        "ccrc_by_cluster": {},
    }

    for label_key in ["y_event", "y_status"]:
        print(f"\n{'-' * 60}\n  Label: {label_key}\n{'-' * 60}")

        # ─── Marginal CRC (for comparison, already reported in §B.12) ───
        marg_fpr_list = []
        ccrc_fpr_list = []
        per_event_detail = []

        for held_id in clusterable_ids:
            # Held-out
            test_scores = np.array(per_event[str(held_id)]["scores"], dtype=np.float64)
            test_labels = np.array(per_event[str(held_id)][label_key], dtype=np.int64)
            if len(np.unique(test_labels)) < 2:
                continue

            held_cluster = cluster_map[held_id]

            # --- MARGINAL CRC: use ALL other events ---
            marg_scores, marg_labels = [], []
            for eid in clusterable_ids:
                if eid == held_id:
                    continue
                marg_scores.extend(per_event[str(eid)]["scores"])
                marg_labels.extend(per_event[str(eid)][label_key])
            marg_scores = np.array(marg_scores)
            marg_labels = np.array(marg_labels)
            if marg_labels.sum() == 0 or marg_labels.sum() == len(marg_labels):
                continue
            tau_marg = marginal_crc_threshold(marg_scores, marg_labels, target_fpr)
            marg_fpr = empirical_fpr(test_scores, test_labels, tau_marg)
            marg_fpr_list.append(marg_fpr)

            # --- CCRC: use only events in SAME causal cluster ---
            cluster_scores, cluster_labels_arr = [], []
            for eid in clusterable_ids:
                if eid == held_id:
                    continue
                if cluster_map[eid] != held_cluster:
                    continue
                cluster_scores.extend(per_event[str(eid)]["scores"])
                cluster_labels_arr.extend(per_event[str(eid)][label_key])
            cluster_scores = np.array(cluster_scores)
            cluster_labels_arr = np.array(cluster_labels_arr)
            if (len(cluster_scores) < 100 or cluster_labels_arr.sum() == 0
                    or cluster_labels_arr.sum() == len(cluster_labels_arr)):
                # Too few cluster peers — fall back to marginal
                tau_ccrc = tau_marg
            else:
                tau_ccrc = marginal_crc_threshold(cluster_scores, cluster_labels_arr,
                                                      target_fpr)
            ccrc_fpr = empirical_fpr(test_scores, test_labels, tau_ccrc)
            ccrc_fpr_list.append(ccrc_fpr)

            per_event_detail.append({
                "event_id": held_id,
                "cluster": int(held_cluster),
                "n_peers_in_cluster": int(sum(1 for eid in clusterable_ids
                                                   if eid != held_id
                                                   and cluster_map[eid] == held_cluster)),
                "marginal_fpr": float(marg_fpr),
                "ccrc_fpr": float(ccrc_fpr),
                "delta_fpr": float(ccrc_fpr - marg_fpr),
            })

        mean_marg = float(np.mean(marg_fpr_list))
        mean_ccrc = float(np.mean(ccrc_fpr_list))
        delta = mean_ccrc - mean_marg

        print(f"  MARGINAL CRC empirical FPR: {mean_marg:.4f}  (target ≤ {target_fpr})")
        print(f"  CCRC (causal-stratified):   {mean_ccrc:.4f}  (target ≤ {target_fpr})")
        print(f"  Δ = {delta:+.4f}   "
              f"{'improved' if delta < 0 else 'worse or flat'}")

        marg_holds = mean_marg <= target_fpr + 0.01
        ccrc_holds = mean_ccrc <= target_fpr + 0.01
        print(f"  Marginal holds: {marg_holds}   |   CCRC holds: {ccrc_holds}")

        results["marginal_crc"][label_key] = {
            "empirical_fpr_mean": mean_marg, "holds": marg_holds,
        }
        results["ccrc_by_cluster"][label_key] = {
            "empirical_fpr_mean": mean_ccrc, "holds": ccrc_holds,
            "delta_vs_marginal": delta, "per_event": per_event_detail,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "causal_conformal.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Saved: docs/results/causal_conformal.json")


if __name__ == "__main__":
    main()
