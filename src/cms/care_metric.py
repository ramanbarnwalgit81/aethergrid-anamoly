"""
Official CARE score (Gück, Roelofs et al. 2024, arXiv:2404.10320 / Data 9(12):138).

CARE = Coverage, Accuracy, Reliability, Earliness. Faithful re-implementation:

  Coverage   F̄β  : Fβ (β=½) of point alerts vs the labelled event window,
                    computed ONLY on normal-status points of *anomaly* datasets,
                    averaged over anomaly datasets.
  Accuracy   Ācc : true-negative rate on normal-status points of *normal*
                    datasets, averaged over normal datasets.
  Reliability EFβ: event-level Fβ. A dataset "raises an alarm" iff a criticality
                    counter (increment on alert, decrement otherwise, floored at
                    0) peaks ≥ ``crit_threshold`` (default 72 ≈ 12 h at 10-min).
  Earliness  W̄S : within each anomaly's event window, alerts in the first half
                    weigh 1, decaying linearly to 0 by the window end;
                    WS = Σ(w·alert)/Σw, averaged over anomaly datasets.

  WA   = (F̄β + W̄S + EFβ + 2·Ācc) / 5
  CARE = 0     if no anomaly events are detected at all,
         Ācc   if Ācc < 0.5,
         WA    otherwise.

Reference points from the paper: Random = 0.5, Isolation Forest ≈ 0.45,
Autoencoder = 0.66 (published "good detector"), perfect = 1.0.

The ``DatasetScore`` inputs carry per-prediction-frame arrays only; thresholds
must be derived from TRAIN data by the caller to keep evaluation leak-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

BETA = 0.5
NORMAL_STATUS = {0, 2}  # README: status 0 and 2 are "normal operation"
CRIT_THRESHOLD = 72  # ≈ 12 h of sustained detection at 10-min resolution
WEIGHTS = (1.0, 1.0, 1.0, 2.0)  # (Coverage, Earliness, Reliability, Accuracy)


@dataclass
class DatasetScore:
    """Per-dataset scoring inputs, aligned to the PREDICTION frame."""

    label: str  # "anomaly" | "normal"
    score: np.ndarray  # anomaly score per prediction-frame row (higher = worse)
    status: np.ndarray  # status_type_id per prediction-frame row
    event_mask: np.ndarray  # True inside the labelled event window
    threshold: float  # alert iff score >= threshold (derived from TRAIN data)

    def alerts(self) -> np.ndarray:
        return np.asarray(self.score) >= self.threshold

    def normal_status_mask(self) -> np.ndarray:
        return np.isin(np.asarray(self.status), list(NORMAL_STATUS))


def _fbeta(tp: float, fp: float, fn: float, beta: float = BETA) -> float:
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return float((1 + b2) * tp / denom) if denom > 0 else 0.0


def _criticality_alarm(alerts: np.ndarray, threshold: int = CRIT_THRESHOLD) -> bool:
    """Increment-on-alert / decrement-otherwise counter; alarm if peak ≥ threshold."""
    crit = 0
    peak = 0
    for a in alerts:
        crit = crit + 1 if a else max(0, crit - 1)
        if crit > peak:
            peak = crit
    return peak >= threshold


def _coverage_one(ds: DatasetScore) -> float:
    ns = ds.normal_status_mask()
    a = ds.alerts() & ns
    ev = np.asarray(ds.event_mask) & ns
    tp = float(np.sum(a & ev))
    fp = float(np.sum(a & ~ev))
    fn = float(np.sum(~a & ev))
    return _fbeta(tp, fp, fn)


def _accuracy_one(ds: DatasetScore) -> float:
    ns = ds.normal_status_mask()
    a = ds.alerts() & ns
    n = float(np.sum(ns))
    if n == 0:
        return 1.0
    fp = float(np.sum(a))
    tn = n - fp
    return tn / (fp + tn) if (fp + tn) > 0 else 1.0


def _earliness_one(ds: DatasetScore) -> float:
    ev = np.asarray(ds.event_mask)
    idx = np.where(ev)[0]
    if idx.size == 0:
        return 0.0
    a = ds.alerts()
    n = idx.size
    # Fractional position within the event window [0, 1).
    frac = (np.arange(n)) / max(n - 1, 1)
    w = np.where(frac <= 0.5, 1.0, np.clip(2.0 * (1.0 - frac), 0.0, 1.0))
    detected = a[idx].astype(float)
    wsum = float(np.sum(w))
    return float(np.sum(w * detected) / wsum) if wsum > 0 else 0.0


def per_dataset_contributions(scores: Sequence[DatasetScore],
                              crit_threshold: int = CRIT_THRESHOLD) -> list:
    """Compact per-dataset scalars sufficient to recompute aggregate CARE on a
    bootstrap resample (enables CARE significance testing without re-running)."""
    out = []
    for ds in scores:
        alarm = _criticality_alarm(ds.alerts(), crit_threshold)
        out.append({
            "label": ds.label,
            "alarm": bool(alarm),
            "coverage": _coverage_one(ds) if ds.label == "anomaly" else None,
            "earliness": _earliness_one(ds) if ds.label == "anomaly" else None,
            "accuracy": _accuracy_one(ds) if ds.label != "anomaly" else None,
        })
    return out


def care_from_contributions(recs, weights=WEIGHTS) -> float:
    """Aggregate CARE from per-dataset contribution records (for bootstrap)."""
    cov = [r["coverage"] for r in recs if r["label"] == "anomaly"]
    ear = [r["earliness"] for r in recs if r["label"] == "anomaly"]
    acc = [r["accuracy"] for r in recs if r["label"] != "anomaly"]
    ev_tp = sum(1 for r in recs if r["label"] == "anomaly" and r["alarm"])
    ev_fn = sum(1 for r in recs if r["label"] == "anomaly" and not r["alarm"])
    ev_fp = sum(1 for r in recs if r["label"] != "anomaly" and r["alarm"])
    F = float(np.mean(cov)) if cov else 0.0
    W = float(np.mean(ear)) if ear else 0.0
    A = float(np.mean(acc)) if acc else 0.0
    EF = _fbeta(ev_tp, ev_fp, ev_fn)
    w1, w2, w3, w4 = weights
    WA = (w1 * F + w2 * W + w3 * EF + w4 * A) / (w1 + w2 + w3 + w4)
    if ev_tp == 0:
        return 0.0
    return A if A < 0.5 else WA


def compute_care(scores: Sequence[DatasetScore],
                 crit_threshold: int = CRIT_THRESHOLD) -> dict:
    """Compute the official CARE score and its components from per-dataset scores."""
    cov, ear = [], []
    acc = []
    ev_tp = ev_fp = ev_fn = 0

    for ds in scores:
        alarm = _criticality_alarm(ds.alerts(), crit_threshold)
        if ds.label == "anomaly":
            cov.append(_coverage_one(ds))
            ear.append(_earliness_one(ds))
            if alarm:
                ev_tp += 1
            else:
                ev_fn += 1
        else:  # normal dataset
            acc.append(_accuracy_one(ds))
            if alarm:
                ev_fp += 1

    F_bar = float(np.mean(cov)) if cov else 0.0
    WS_bar = float(np.mean(ear)) if ear else 0.0
    Acc_bar = float(np.mean(acc)) if acc else 0.0
    EF_beta = _fbeta(ev_tp, ev_fp, ev_fn)

    w1, w2, w3, w4 = WEIGHTS
    WA = (w1 * F_bar + w2 * WS_bar + w3 * EF_beta + w4 * Acc_bar) / (w1 + w2 + w3 + w4)

    if ev_tp == 0:
        care = 0.0
    elif Acc_bar < 0.5:
        care = Acc_bar
    else:
        care = WA

    return {
        "care_score": round(care, 4),
        "WA": round(WA, 4),
        "coverage_fbeta": round(F_bar, 4),
        "earliness_ws": round(WS_bar, 4),
        "reliability_efbeta": round(EF_beta, 4),
        "accuracy": round(Acc_bar, 4),
        "event_tp": ev_tp,
        "event_fp": ev_fp,
        "event_fn": ev_fn,
        "n_anomaly": len(cov),
        "n_normal": len(acc),
        "crit_threshold": crit_threshold,
        "beta": BETA,
    }


# --------------------------------------------------------------------------
# Self-test: perfect / random / silent detectors hit the documented anchors.
# --------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)

    def make(label, kind):
        n = 300
        status = np.zeros(n, dtype=int)  # all normal status
        event = np.zeros(n, dtype=bool)
        if label == "anomaly":
            event[150:] = True  # event in 2nd half of prediction frame
        if kind == "perfect":
            score = event.astype(float)  # 1 inside event, 0 outside
            thr = 0.5
        elif kind == "silent":
            score = np.zeros(n)
            thr = 0.5
        else:  # random
            score = rng.random(n)
            thr = 0.5
        return DatasetScore(label, score, status, event, thr)

    for kind in ["perfect", "silent", "random"]:
        dss = [make("anomaly", kind) for _ in range(5)] + \
              [make("normal", kind) for _ in range(5)]
        r = compute_care(dss, crit_threshold=72)
        print(f"{kind:8s} -> CARE={r['care_score']:.3f}  "
              f"cov={r['coverage_fbeta']:.2f} ws={r['earliness_ws']:.2f} "
              f"ef={r['reliability_efbeta']:.2f} acc={r['accuracy']:.2f} "
              f"(tp={r['event_tp']} fp={r['event_fp']} fn={r['event_fn']})")


if __name__ == "__main__":
    _selftest()
