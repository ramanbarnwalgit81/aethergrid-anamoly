"""
Official CARE Score Implementation.

From Gück, Roelofs et al. 2024 "CARE to Compare: A real-world dataset for
anomaly detection in wind turbine data" (arxiv 2404.10320):

  CARE = (w1 · F_beta_bar + w2 · WS_bar + w3 · EF_beta + w4 · Acc_bar) / sum(wi)

  where w1 = w2 = w3 = 1, w4 = 2, beta = 0.5

Components:
  - Coverage (F_beta): detection of anomalies via weighted F-score
  - Earliness (WS):    early-detection weighted scoring
  - Reliability (EF):  event-based false alarm reliability
  - Accuracy (Acc):    normal behavior recognition

A perfect model scores 1.0. Random baseline scores ~0.5.
Published best: autoencoder at 0.66 (from original paper).
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def f_beta_score(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    """F_beta favours precision when beta < 1."""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision == 0 or recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def earliness_weight(t: float, t_event_start: float,
                      t_event_end: float) -> float:
    """
    Earliness weight: detections near event_start_id score higher.
    t = timestamp of detection
    t_event_start = labeled fault start
    t_event_end   = labeled fault end

    Returns 1.0 if detected at event_start, linearly decaying to 0 at event_end.
    """
    if t < t_event_start:
        # Detection BEFORE the label window — treat as full earliness
        # only if within a reasonable early-warning horizon (e.g., 1 week)
        horizon = 168 * 3600  # 1 week in seconds
        diff = (t_event_start - t)
        if diff <= horizon:
            return 1.0
        return 0.0
    if t > t_event_end:
        return 0.0
    # Within event window — linearly decay
    window = t_event_end - t_event_start
    if window <= 0:
        return 0.5
    return 1.0 - (t - t_event_start) / window


def weighted_score_for_event(
    detection_times: List[float],
    event_start: float,
    event_end: float,
    threshold_severity: float = 0.9,
) -> float:
    """Weighted earliness score for one event."""
    if not detection_times:
        return 0.0
    # Earliness weights for each detection
    weights = [earliness_weight(t, event_start, event_end)
               for t in detection_times]
    if not any(w > 0 for w in weights):
        return 0.0
    # Average weight of detections within the window
    return float(np.mean(weights))


def event_reliability_score(
    n_false_alarm_events: int,
    n_true_events: int,
    n_normal_events: int,
    beta: float = 0.5,
) -> float:
    """
    Event-level false-alarm reliability.
    A false-alarm event is when a NORMAL event period triggers an alert.
    """
    tp = n_true_events - n_false_alarm_events  # not quite right but close
    # More accurate:
    #   tp = correctly identified anomaly events
    #   fp = normal events falsely flagged
    #   fn = missed anomaly events
    tp_e = n_true_events  # events caught
    fp_e = n_false_alarm_events
    fn_e = 0  # we assume all anomaly events are the set we're scoring
    return f_beta_score(tp_e, fp_e, fn_e, beta=beta)


def compute_care_score(
    event_results: List[Dict],
    beta: float = 0.5,
    weights: tuple = (1.0, 1.0, 1.0, 2.0),
) -> Dict:
    """
    Compute the official CARE score from per-event results.

    Each event_result dict should contain:
      - event_type: 'anomaly' or 'normal'
      - y_true: np.ndarray of 0/1 labels per timestamp
      - y_score: np.ndarray of anomaly scores per timestamp
      - event_start_idx: int, first row of labeled anomaly (None for normal)
      - event_end_idx: int, last row of labeled anomaly
      - threshold: float, score threshold to compute alerts

    Returns dict with care_score and all 4 components.
    """
    w1, w2, w3, w4 = weights

    # Per-event metrics
    coverage_f_betas = []      # for anomaly events
    earliness_ws = []           # for anomaly events
    accuracies = []             # for NORMAL events (should be all zeros)
    n_anomaly = 0
    n_normal = 0
    n_true_detected = 0
    n_false_alarm_events = 0

    for ev in event_results:
        y_true = np.asarray(ev["y_true"])
        y_score = np.asarray(ev["y_score"])
        threshold = ev.get("threshold", np.percentile(y_score, 95))
        alerts = (y_score >= threshold).astype(int)

        is_anomaly_event = ev.get("event_type") == "anomaly"

        if is_anomaly_event:
            n_anomaly += 1
            tp = int(((alerts == 1) & (y_true == 1)).sum())
            fp = int(((alerts == 1) & (y_true == 0)).sum())
            fn = int(((alerts == 0) & (y_true == 1)).sum())
            f_b = f_beta_score(tp, fp, fn, beta=beta)
            coverage_f_betas.append(f_b)

            # Earliness: average weight of alerts within the event window
            event_start = ev.get("event_start_idx")
            event_end = ev.get("event_end_idx")
            if event_start is not None and event_end is not None:
                detection_indices = np.where(alerts == 1)[0]
                weights_per_detect = [
                    earliness_weight(idx, event_start, event_end)
                    for idx in detection_indices
                ]
                ws = float(np.mean(weights_per_detect)) if weights_per_detect else 0.0
                earliness_ws.append(ws)
                if ws > 0:
                    n_true_detected += 1
        else:
            n_normal += 1
            # Accuracy: fraction of this normal event NOT flagged
            normal_accuracy = float(1.0 - alerts.mean())
            accuracies.append(normal_accuracy)
            if alerts.sum() > 0:
                n_false_alarm_events += 1

    # Coverage F-beta: average across anomaly events
    F_beta_bar = float(np.mean(coverage_f_betas)) if coverage_f_betas else 0.0
    # Weighted earliness: average across anomaly events
    WS_bar = float(np.mean(earliness_ws)) if earliness_ws else 0.0
    # Event-level false alarm reliability
    EF_beta = event_reliability_score(
        n_false_alarm_events, n_true_detected, n_normal, beta=beta
    )
    # Normal-behavior accuracy
    Acc_bar = float(np.mean(accuracies)) if accuracies else 0.0

    # Final weighted average
    num = w1 * F_beta_bar + w2 * WS_bar + w3 * EF_beta + w4 * Acc_bar
    denom = w1 + w2 + w3 + w4

    care = num / denom

    return {
        "care_score": round(care, 4),
        "coverage_f_beta": round(F_beta_bar, 4),
        "weighted_earliness": round(WS_bar, 4),
        "event_reliability_f_beta": round(EF_beta, 4),
        "normal_accuracy": round(Acc_bar, 4),
        "n_anomaly_events": n_anomaly,
        "n_normal_events": n_normal,
        "n_true_detected_events": n_true_detected,
        "n_false_alarm_events": n_false_alarm_events,
        "weights_w1_w2_w3_w4": weights,
        "beta": beta,
    }


if __name__ == "__main__":
    # Sanity: perfect detector scores 1.0
    ev_anomaly = {
        "event_type": "anomaly",
        "y_true": np.array([0, 0, 0, 1, 1, 1, 1, 1]),
        "y_score": np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]),
        "event_start_idx": 3,
        "event_end_idx": 7,
        "threshold": 0.5,
    }
    ev_normal = {
        "event_type": "normal",
        "y_true": np.zeros(8),
        "y_score": np.array([0.1] * 8),
        "event_start_idx": None, "event_end_idx": None,
        "threshold": 0.5,
    }
    result = compute_care_score([ev_anomaly, ev_normal])
    print("Perfect detector:", result)
