"""Weighted split-conformal (Tibshirani et al. 2019) on the CARE hard task.

Marginal split-conformal assumes exchangeability between calibration and test
samples. On the event-window label this fails because each event's score
distribution is different — the calibration quantile is not a valid threshold
for the held-out event (Section V.E of the paper, empirical FPR 0.158 vs
target 0.05).

Tibshirani et al. 2019 (NeurIPS) show that when calibration and test come from
different distributions P and Q with a likelihood-ratio weighting w(x) = dQ/dP,
split-conformal recovers marginal coverage by taking the weighted quantile of
the calibration nonconformity scores:

    tau = inf{ t : sum_i w_i * 1{s_i > t} / sum_i w_i <= alpha }

We approximate dQ/dP on each held-out event by training a simple logistic
regression to distinguish calibration-set rows from held-out-event rows using
the physics residuals, rolling score statistics, and turbine / operating-state
features. The weights are clipped to [0.1, 10] to avoid a single outlier
dominating the quantile.

Usage:  python -m src.benchmark.weighted_conformal
Outputs: docs/results/weighted_conformal.json
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("docs/results")
ALPHA = 0.05  # target FPR
BETA = 0.20   # target FNR


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Empirical weighted (1 - q) quantile."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cw, 1 - q, side="left").clip(0, len(v) - 1)])


def estimate_weights(cal_X: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    """Logistic-regression density ratio w(x) = P(test|x)/P(cal|x) * (n_cal/n_test)."""
    X = np.vstack([cal_X, test_X])
    y = np.concatenate([np.zeros(len(cal_X)), np.ones(len(test_X))])
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y)
    p_test = clf.predict_proba(sc.transform(cal_X))[:, 1]
    p_cal = 1.0 - p_test
    w = (p_test / np.clip(p_cal, 1e-4, None)) * (len(cal_X) / max(len(test_X), 1))
    return np.clip(w, 0.1, 10.0)


def run_loeo_weighted_crc(label_key: str = "y_event") -> dict:
    """Leave-one-event-out weighted CRC on CARE Farm A v7 scores."""
    with (RESULTS_DIR / "care_ensemble_v7_per_event_scores.json").open() as f:
        v7 = json.load(f)

    event_ids = sorted(int(k) for k in v7.keys())
    # Features for weight estimation: score, rolling_mean_10, rolling_std_10,
    # score_delta, rolling_rank
    def feature_matrix(scores: np.ndarray) -> np.ndarray:
        s = pd.Series(scores)
        return np.column_stack([
            scores,
            s.rolling(10, min_periods=1).mean().values,
            s.rolling(10, min_periods=1).std().fillna(0).values,
            s.diff().fillna(0).values,
            s.rolling(100, min_periods=1).rank(pct=True).fillna(0.5).values,
        ])

    results_marginal = []
    results_weighted = []
    for held in event_ids:
        record = v7[str(held)]
        y_h = np.asarray(record.get(label_key, []), dtype=int)
        s_h = np.asarray(record["scores"], dtype=float)
        if len(y_h) == 0 or len(np.unique(y_h)) < 2:
            continue
        # Calibration pool: all other events (both classes)
        cal_scores_neg, cal_scores_pos = [], []
        cal_feat_neg, cal_feat_pos = [], []
        for other in event_ids:
            if other == held:
                continue
            r = v7[str(other)]
            y_o = np.asarray(r.get(label_key, []), dtype=int)
            s_o = np.asarray(r["scores"], dtype=float)
            F_o = feature_matrix(s_o)
            cal_scores_neg.append(s_o[y_o == 0])
            cal_scores_pos.append(s_o[y_o == 1])
            cal_feat_neg.append(F_o[y_o == 0])
            cal_feat_pos.append(F_o[y_o == 1])
        if not cal_scores_neg:
            continue
        s_cal_neg = np.concatenate(cal_scores_neg)
        s_cal_pos = np.concatenate(cal_scores_pos)
        F_cal_neg = np.vstack(cal_feat_neg)
        F_cal_pos = np.vstack(cal_feat_pos)

        F_test = feature_matrix(s_h)

        # --- Marginal (unweighted) ---
        n = len(s_cal_neg)
        q_level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
        tau_fpr_m = float(np.quantile(s_cal_neg, q_level))
        tau_fnr_m = float(np.quantile(s_cal_pos, BETA))
        pred_m = (s_h >= tau_fpr_m).astype(int)
        fpr_m = float(np.mean((pred_m == 1) & (y_h == 0))) / max(1e-9, (y_h == 0).mean())
        fnr_m = float(np.mean((pred_m == 0) & (y_h == 1))) / max(1e-9, (y_h == 1).mean())

        # --- Weighted ---
        w_neg = estimate_weights(F_cal_neg, F_test)
        w_pos = estimate_weights(F_cal_pos, F_test)
        tau_fpr_w = weighted_quantile(s_cal_neg, w_neg, ALPHA)
        tau_fnr_w = weighted_quantile(s_cal_pos, w_pos, 1 - BETA)
        pred_w = (s_h >= tau_fpr_w).astype(int)
        fpr_w = float(np.mean((pred_w == 1) & (y_h == 0))) / max(1e-9, (y_h == 0).mean())
        fnr_w = float(np.mean((pred_w == 0) & (y_h == 1))) / max(1e-9, (y_h == 1).mean())

        results_marginal.append({"event_id": held, "fpr": fpr_m, "fnr": fnr_m})
        results_weighted.append({"event_id": held, "fpr": fpr_w, "fnr": fnr_w})

    def aggregate(rs):
        return {
            "n_events": len(rs),
            "mean_fpr": round(float(np.mean([r["fpr"] for r in rs])), 4),
            "mean_fnr": round(float(np.mean([r["fnr"] for r in rs])), 4),
            "p_events_fpr_above_target": round(
                float(np.mean([r["fpr"] > ALPHA for r in rs])), 3
            ),
        }

    return {
        "label": label_key,
        "target_alpha_fpr": ALPHA,
        "target_beta_fnr": BETA,
        "marginal_CRC": aggregate(results_marginal),
        "weighted_CRC_Tibshirani2019": aggregate(results_weighted),
        "per_event_marginal": results_marginal,
        "per_event_weighted": results_weighted,
    }


def main():
    out = {
        "y_event":     run_loeo_weighted_crc("y_event"),
        "y_precursor": run_loeo_weighted_crc("y_precursor"),
        "y_status":    run_loeo_weighted_crc("y_status"),
    }
    (RESULTS_DIR / "weighted_conformal.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{'Label':<15} {'Marginal FPR':>14} {'Weighted FPR':>14} {'Marg FNR':>10} {'Wt FNR':>10}")
    print("-" * 75)
    for lab, r in out.items():
        m = r["marginal_CRC"]; w = r["weighted_CRC_Tibshirani2019"]
        print(f"{lab:<15} {m['mean_fpr']:>14.4f} {w['mean_fpr']:>14.4f} "
              f"{m['mean_fnr']:>10.4f} {w['mean_fnr']:>10.4f}")
    print(f"\n[OK] Saved: docs/results/weighted_conformal.json")


if __name__ == "__main__":
    main()
