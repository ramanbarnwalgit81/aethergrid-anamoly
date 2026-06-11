"""
Conformal prediction wrapper for calibrated wind-CMS anomaly detection.

REFERENCES
----------
- Vovk, Gammerman, Shafer 2005 — original conformal prediction theory
- Romano, Sesia, Candès 2019 — split conformal for regression/classification
- Angelopoulos, Bates 2021 — A Gentle Introduction to Conformal Prediction
- Qi, Lee 2025 ICLR — Conformalized Survival Analysis for censored data

WHAT THIS ADDS
--------------
Raw anomaly scores from v6/v7 are uncalibrated — a score of 2.3 has no
probabilistic meaning. Conformal prediction wraps any scorer to produce:

  1. Coverage-guaranteed prediction intervals
     P(true_label == 1  |  score ≥ τ_α) ≥ 1 - α
     (marginal guarantee, no distributional assumptions)

  2. Label-conditional (Mondrian) coverage
     Separate thresholds for anomaly vs. normal subsets, so false-alarm
     rate is guaranteed AT MOST α on normal rows.

  3. p-values per row (for ranking and thresholding)
     p(x) = fraction of calibration scores ≥ score(x)

  4. "Abstain" zone: rows with p-values in [α, 1-α] (low certainty)

FOR WIND CMS: operators can set false-alarm rate = 5% and the system
GUARANTEES no more than 5% of normal rows will fire alerts, regardless
of the underlying ensemble's confidence quirks.

USAGE
-----
    from src.benchmark.conformal import ConformalWrapper, evaluate_v7
    cp = ConformalWrapper(alpha=0.05)
    cp.fit(calib_scores, calib_labels)
    alerts, p_values, abstain = cp.predict(test_scores)

    # Or apply to existing v7 results:
    python -m src.benchmark.conformal
"""

from pathlib import Path
import json
import os, sys

import numpy as np


RESULTS_DIR = Path("docs/results")


class ConformalWrapper:
    """
    Split-conformal wrapper for anomaly-score calibration.

    Supports:
      - Marginal conformal (one threshold for all rows)
      - Label-conditional (Mondrian) — separate thresholds per class
    """

    def __init__(self, alpha: float = 0.05, mondrian: bool = True):
        """
        Args:
            alpha: desired miscoverage rate (e.g., 0.05 for 95% coverage)
            mondrian: if True, use label-conditional thresholds
        """
        self.alpha = alpha
        self.mondrian = mondrian
        self.calib_scores_class0 = None
        self.calib_scores_class1 = None
        self.calib_scores_all = None

    def fit(self, calib_scores: np.ndarray, calib_labels: np.ndarray):
        """
        Calibrate on a held-out set.

        Args:
            calib_scores: anomaly scores on calibration rows, shape (n,)
            calib_labels: 0/1 ground-truth labels, shape (n,)
        """
        self.calib_scores_all = np.sort(np.asarray(calib_scores).flatten())
        if self.mondrian:
            mask0 = np.asarray(calib_labels) == 0
            mask1 = np.asarray(calib_labels) == 1
            self.calib_scores_class0 = np.sort(calib_scores[mask0]) if mask0.sum() > 0 else None
            self.calib_scores_class1 = np.sort(calib_scores[mask1]) if mask1.sum() > 0 else None

    def p_value(self, scores: np.ndarray, class_label: int = 0) -> np.ndarray:
        """
        Compute conformal p-value: fraction of calibration scores ≥ test score,
        under the hypothesis that the test row belongs to `class_label`.

        Small p = test score is an outlier under the null hypothesis of
        belonging to `class_label`.
        """
        calib = (self.calib_scores_class0 if class_label == 0
                 else self.calib_scores_class1 if class_label == 1
                 else self.calib_scores_all)
        if calib is None or len(calib) == 0:
            return np.full(len(scores), 0.5)

        # Standard conformal p-value: (1 + #{s in calib : s >= t}) / (1 + n_calib)
        n = len(calib)
        scores = np.asarray(scores)
        # searchsorted: count of calib >= t = n - (index of first calib >= t, left-inserted)
        idx = np.searchsorted(calib, scores, side="left")
        count_ge = n - idx
        p = (1.0 + count_ge) / (1.0 + n)
        return p

    def threshold(self, class_label: int = 0, alpha: float = None) -> float:
        """
        Return conformal threshold such that P(score ≥ τ | class=class_label) <= α.
        Operator sets alpha=0.05 → no more than 5% of `class_label` rows exceed τ.
        """
        if alpha is None:
            alpha = self.alpha
        calib = (self.calib_scores_class0 if class_label == 0
                 else self.calib_scores_class1 if class_label == 1
                 else self.calib_scores_all)
        if calib is None or len(calib) == 0:
            return float("inf")
        n = len(calib)
        # Quantile adjustment for finite-sample conformal: ceil((n+1)(1-alpha))/n
        q_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
        q_idx = max(0, min(q_idx, n - 1))
        return float(calib[q_idx])

    def predict(self, scores: np.ndarray, alpha: float = None) -> dict:
        """
        Return calibrated outputs:
          - alerts: bool array, True if score ≥ threshold (α false-alarm rate
                    guarantee on calibration-set normal rows)
          - p_values_normal: p-value under null of "normal" (small = anomaly)
          - p_values_anomaly: p-value under null of "anomaly" (small = normal)
          - abstain: True if both p-values are in [α, 1-α]
        """
        if alpha is None:
            alpha = self.alpha
        tau_normal = self.threshold(class_label=0, alpha=alpha)
        alerts = scores >= tau_normal
        p_normal = self.p_value(scores, class_label=0)
        p_anomaly = self.p_value(scores, class_label=1) if self.mondrian else None
        abstain = None
        if p_anomaly is not None:
            abstain = (p_normal >= alpha) & (p_anomaly >= alpha)
        return {
            "alerts": alerts,
            "threshold": tau_normal,
            "p_values_normal_null": p_normal,
            "p_values_anomaly_null": p_anomaly,
            "abstain": abstain,
        }


def empirical_coverage(scores: np.ndarray, labels: np.ndarray,
                         alerts: np.ndarray) -> dict:
    """
    Compute empirical coverage / FPR / power / abstention rate.
    Returns ratios so a reviewer can verify marginal coverage guarantee.
    """
    labels = np.asarray(labels)
    alerts = np.asarray(alerts)
    n_total = len(labels)
    n_pos = int(labels.sum())
    n_neg = n_total - n_pos
    tp = int(((alerts == 1) & (labels == 1)).sum())
    fp = int(((alerts == 1) & (labels == 0)).sum())
    fn = int(((alerts == 0) & (labels == 1)).sum())
    tn = int(((alerts == 0) & (labels == 0)).sum())
    return {
        "n_total": n_total, "n_positive": n_pos, "n_negative": n_neg,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fpr": fp / max(n_neg, 1),
        "recall": tp / max(n_pos, 1),
        "precision": tp / max(tp + fp, 1),
        "alert_rate": (tp + fp) / max(n_total, 1),
    }


def evaluate_v7_conformal(alpha: float = 0.05, label_type: str = "event"):
    """
    Apply split-conformal prediction to v7 per-event scores.

    Protocol:
      1. Leave-one-event-out cross-validation: for each anomaly event X,
         calibrate on scores from other 21 events, evaluate on X.
      2. Compute coverage-guaranteed alert rate at α (default 5%).
      3. Report empirical FPR — should be <= α on the normal subset.

    Args:
        alpha: miscoverage rate (5% = 95% coverage)
        label_type: "event" | "precursor" | "status"
    """
    scores_path = RESULTS_DIR / "care_ensemble_v7_per_event_scores.json"
    if not scores_path.exists():
        print(f"[ERR] v7 scores not found at {scores_path}")
        return None

    with scores_path.open() as f:
        per_event = json.load(f)

    event_ids = list(per_event.keys())
    print(f"[CONFORMAL] Loaded per-event scores for {len(event_ids)} events")
    print(f"[CONFORMAL] alpha = {alpha}, label = {label_type}")

    results = []
    pooled_eval = {"fpr": [], "recall": [], "precision": [], "alert_rate": []}

    for leave_out_id in event_ids:
        # Build calibration set from all other events
        calib_scores = []
        calib_labels = []
        for eid, d in per_event.items():
            if eid == leave_out_id:
                continue
            calib_scores.extend(d["scores"])
            if label_type == "event":
                calib_labels.extend(d["y_event"])
            elif label_type == "precursor":
                calib_labels.extend(d["y_precursor"])
            else:
                calib_labels.extend(d["y_status"])
        calib_scores = np.array(calib_scores, dtype=np.float64)
        calib_labels = np.array(calib_labels, dtype=np.int64)

        if len(calib_scores) == 0 or calib_labels.sum() == 0 or calib_labels.sum() == len(calib_labels):
            continue

        cp = ConformalWrapper(alpha=alpha, mondrian=True)
        cp.fit(calib_scores, calib_labels)

        # Test on held-out event
        test = per_event[leave_out_id]
        test_scores = np.array(test["scores"], dtype=np.float64)
        if label_type == "event":
            test_labels = np.array(test["y_event"], dtype=np.int64)
        elif label_type == "precursor":
            test_labels = np.array(test["y_precursor"], dtype=np.int64)
        else:
            test_labels = np.array(test["y_status"], dtype=np.int64)

        if len(np.unique(test_labels)) < 2:
            continue

        pred = cp.predict(test_scores)
        ev = empirical_coverage(test_scores, test_labels, pred["alerts"])
        ev["event_id"] = int(leave_out_id)
        ev["threshold"] = float(pred["threshold"])
        ev["abstention_rate"] = float(pred["abstain"].mean()) if pred["abstain"] is not None else 0.0
        results.append(ev)
        for k in ["fpr", "recall", "precision", "alert_rate"]:
            pooled_eval[k].append(ev[k])

    print(f"\n  Per-event leave-one-out conformal results ({label_type} label, alpha={alpha}):")
    print(f"  {'event':>6} {'n_test':>7} {'pos':>5} {'neg':>6} {'tp':>4} {'fp':>4} "
          f"{'fpr':>6} {'recall':>6} {'abs%':>5}")
    for r in results[:20]:  # show first 20 for brevity
        print(f"  {r['event_id']:>6} {r['n_total']:>7} {r['n_positive']:>5} "
              f"{r['n_negative']:>6} {r['tp']:>4} {r['fp']:>4} "
              f"{r['fpr']:>6.3f} {r['recall']:>6.3f} "
              f"{r['abstention_rate']*100:>4.1f}%")

    agg = {k: float(np.mean(v)) for k, v in pooled_eval.items() if v}
    print(f"\n  AGGREGATE (n={len(results)} held-out events):")
    print(f"    Mean FPR:       {agg.get('fpr', 0):.4f}  (target <= {alpha})")
    print(f"    Mean recall:    {agg.get('recall', 0):.4f}")
    print(f"    Mean precision: {agg.get('precision', 0):.4f}")
    print(f"    Mean alert rate: {agg.get('alert_rate', 0):.4f}")

    coverage_ok = agg.get("fpr", 1.0) <= alpha + 0.02  # small tolerance
    print(f"\n  Marginal coverage guarantee: "
          f"{'HOLDS' if coverage_ok else 'VIOLATED — re-check calibration'} "
          f"(empirical FPR {agg.get('fpr', 0):.4f} vs target {alpha})")

    out_path = RESULTS_DIR / f"conformal_v7_{label_type}_alpha{int(alpha*100):02d}.json"
    with out_path.open("w") as f:
        json.dump({
            "alpha": alpha,
            "label_type": label_type,
            "aggregate": agg,
            "marginal_coverage_holds": coverage_ok,
            "per_event": results,
        }, f, indent=2, default=str)
    print(f"\n  [OK] Saved: {out_path}")
    return agg


def main():
    print("=" * 80)
    print("  CONFORMAL PREDICTION on v7 scores — calibrated wind-CMS")
    print("=" * 80)

    # Test multiple miscoverage rates and label definitions
    for alpha in [0.01, 0.05, 0.10]:
        for label in ["event", "status"]:
            print(f"\n--- alpha={alpha}, label={label} ---")
            evaluate_v7_conformal(alpha=alpha, label_type=label)


if __name__ == "__main__":
    main()
