"""
WindBench Unified Metrics.

All metrics used across the benchmark, implemented once with consistent
definitions. Every model's results go through these functions.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, mean_absolute_error, mean_squared_error, confusion_matrix,
)
from scipy.stats import spearmanr


def detection_metrics(y_true: np.ndarray, y_score: np.ndarray,
                      threshold: float = 0.5) -> Dict:
    """Core detection metrics + bootstrap-ready."""
    if len(np.unique(y_true)) < 2:
        return {"error": "single class", "n": len(y_true)}

    # AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_roc = float(auc(fpr, tpr))
    auc_pr = float(average_precision_score(y_true, y_score))

    # Recall at specific FPRs
    def _recall_at_fpr(target):
        idx = np.where(fpr <= target)[0]
        return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

    # Best F1 across thresholds
    prec, rec, pr_thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best_f1 = float(f1.max())

    # At operating threshold
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision_op = tp / max(tp + fp, 1)
    recall_op = tp / max(tp + fn, 1)
    f1_op = 2 * precision_op * recall_op / max(precision_op + recall_op, 1e-9)
    fpr_op = fp / max(fp + tn, 1)

    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "recall_at_fpr_0.05": round(_recall_at_fpr(0.05), 4),
        "recall_at_fpr_0.10": round(_recall_at_fpr(0.10), 4),
        "recall_at_fpr_0.15": round(_recall_at_fpr(0.15), 4),
        "f1_best": round(best_f1, 4),
        "precision_at_threshold": round(float(precision_op), 4),
        "recall_at_threshold": round(float(recall_op), 4),
        "fpr_at_threshold": round(float(fpr_op), 4),
        "f1_at_threshold": round(float(f1_op), 4),
        "threshold_used": threshold,
    }


def bootstrap_ci(y_true, y_score, metric_fn, n_boot: int = 1000,
                 seed: int = 42, ci: float = 0.95) -> Dict:
    """Return (mean, CI lower, CI upper) for metric_fn."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    samples = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        try:
            v = metric_fn(y_true[s], y_score[s])
            if np.isfinite(v):
                samples.append(v)
        except Exception:
            continue
    samples = np.array(samples)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(samples)),
        "ci_lower": float(np.percentile(samples, alpha * 100)),
        "ci_upper": float(np.percentile(samples, (1 - alpha) * 100)),
        "std": float(np.std(samples)),
    }


def detection_metrics_with_ci(y_true, y_score, n_boot=1000) -> Dict:
    """Detection metrics + 95% bootstrap confidence intervals."""
    base = detection_metrics(y_true, y_score)

    def _auc(yt, ys):
        if len(np.unique(yt)) < 2:
            return 0.5
        return auc(*roc_curve(yt, ys)[:2])

    def _auc_pr(yt, ys):
        if len(np.unique(yt)) < 2:
            return 0.0
        return average_precision_score(yt, ys)

    def _recall_10(yt, ys):
        if len(np.unique(yt)) < 2:
            return 0.0
        f, t, _ = roc_curve(yt, ys)
        idx = np.where(f <= 0.10)[0]
        return float(t[idx[-1]]) if len(idx) > 0 else 0.0

    base["auc_roc_ci"] = bootstrap_ci(y_true, y_score, _auc, n_boot)
    base["auc_pr_ci"] = bootstrap_ci(y_true, y_score, _auc_pr, n_boot)
    base["recall_at_fpr_0.10_ci"] = bootstrap_ci(y_true, y_score, _recall_10, n_boot)
    return base


def rul_metrics(y_true_rul: np.ndarray, y_pred_rul: np.ndarray) -> Dict:
    """RUL prediction metrics (C-MAPSS-style)."""
    if len(y_true_rul) == 0 or len(y_pred_rul) == 0:
        return {"error": "empty arrays"}

    mae = float(mean_absolute_error(y_true_rul, y_pred_rul))
    rmse = float(np.sqrt(mean_squared_error(y_true_rul, y_pred_rul)))

    within_24h = float(np.mean(np.abs(y_true_rul - y_pred_rul) < 24))
    within_48h = float(np.mean(np.abs(y_true_rul - y_pred_rul) < 48))

    try:
        rho, _ = spearmanr(y_true_rul, y_pred_rul)
        rho = float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        rho = 0.0

    # C-MAPSS scoring: asymmetric penalty (late predictions cost more)
    def cmapss_score(y_true, y_pred):
        diff = y_pred - y_true  # positive = predicted LATE (bad)
        score = np.where(diff < 0,
                         np.exp(-diff / 13.0) - 1,
                         np.exp(diff / 10.0) - 1)
        return float(np.sum(score))

    return {
        "n_samples": int(len(y_true_rul)),
        "rul_mae_hours": round(mae, 2),
        "rul_rmse_hours": round(rmse, 2),
        "rul_within_24h_fraction": round(within_24h, 4),
        "rul_within_48h_fraction": round(within_48h, 4),
        "rul_rho_spearman": round(rho, 4),
        "rul_cmapss_score": round(cmapss_score(y_true_rul, y_pred_rul), 2),
        "rul_mean_true": round(float(y_true_rul.mean()), 2),
        "rul_mean_predicted": round(float(y_pred_rul.mean()), 2),
    }


def transfer_gap(in_domain_auc: float, out_domain_auc: float) -> Dict:
    """Cross-site generalization gap."""
    return {
        "in_domain_auc_roc": round(in_domain_auc, 4),
        "out_domain_auc_roc": round(out_domain_auc, 4),
        "transfer_gap": round(in_domain_auc - out_domain_auc, 4),
        "relative_degradation_pct": round(
            (in_domain_auc - out_domain_auc) / max(in_domain_auc, 1e-9) * 100, 2
        ),
    }


def adversarial_robustness(baseline_detection_rate: float,
                             attacked_detection_rate: float,
                             defended_detection_rate: float) -> Dict:
    """Adversarial attack + defense metrics."""
    return {
        "baseline_detection_rate": round(baseline_detection_rate, 4),
        "attacked_detection_rate": round(attacked_detection_rate, 4),
        "defended_detection_rate": round(defended_detection_rate, 4),
        "attack_success_rate": round(1 - attacked_detection_rate, 4),
        "defense_improvement": round(
            defended_detection_rate - attacked_detection_rate, 4
        ),
    }


def early_warning_metrics(detections_per_fault: list) -> Dict:
    """
    Aggregate early-warning stats across multiple fault events.

    detections_per_fault: list of dicts with:
        'time_to_detection_hours' (hours after fault start — lower is better)
        'lead_time_hours' (hours before catastrophic end — higher is better)
        'detected' (bool)
    """
    detected = [d for d in detections_per_fault if d.get("detected")]
    if not detected:
        return {
            "n_faults": len(detections_per_fault),
            "n_detected": 0,
            "detection_rate": 0.0,
        }

    ttd = [d["time_to_detection_hours"] for d in detected
           if d.get("time_to_detection_hours") is not None]
    lead = [d.get("lead_time_hours", 0) for d in detected]

    return {
        "n_faults": len(detections_per_fault),
        "n_detected": len(detected),
        "detection_rate": round(len(detected) / len(detections_per_fault), 4),
        "ttd_hours_mean": round(float(np.mean(ttd)), 2) if ttd else None,
        "ttd_hours_median": round(float(np.median(ttd)), 2) if ttd else None,
        "ttd_hours_p95": round(float(np.percentile(ttd, 95)), 2) if ttd else None,
        "lead_time_hours_mean": round(float(np.mean(lead)), 2) if lead else None,
        "early_warning_fraction": round(
            float(np.mean([1.0 if d.get("time_to_detection_hours", 999) < 1.0 else 0.0
                           for d in detected])), 4),
    }


def care_score(y_true: np.ndarray, y_score: np.ndarray,
               threshold: float = 0.9, early_bonus_hours: int = 168) -> Dict:
    """
    CARE score from Kreutz et al. 2024.
    Rewards early detection, penalizes false positives.
    """
    alerts = (y_score >= threshold).astype(int)
    tp = int(((alerts == 1) & (y_true == 1)).sum())
    fp = int(((alerts == 1) & (y_true == 0)).sum())
    fn = int(((alerts == 0) & (y_true == 1)).sum())

    score = tp - 0.1 * fp
    if tp > 0:
        score += early_bonus_hours * 0.01
    max_possible = y_true.sum() + early_bonus_hours * 0.01
    if max_possible == 0:
        return {"care_score": 0.0}
    return {
        "care_score": round(float(score / max_possible), 4),
        "tp": tp, "fp": fp, "fn": fn,
    }
