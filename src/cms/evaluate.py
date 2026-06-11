"""
Evaluation harness for the official CARE protocol.

For each dataset of a farm, independently:
  1. fit a Preprocessor + scorer on the TRAIN frame (normal history),
  2. score TRAIN and PREDICTION frames,
  3. set the alert threshold from a fixed quantile of TRAIN scores (leak-free),
  4. build a DatasetScore for the CARE metric.

We report BOTH:
  * the official CARE score (threshold-dependent, the leaderboard metric), and
  * threshold-free event-level discrimination (AUC/AP) where each dataset's
    anomaly score = peak of an EWMA-smoothed score, label = anomaly vs normal.

Nothing is fabricated: results are written to docs/results/cms/ as JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.cms import data as D
from src.cms.care_metric import DatasetScore, compute_care
from src.cms.models import make_scorer
from src.cms.preprocess import Preprocessor

RESULTS_DIR = Path("docs/results/cms")
TRAIN_QUANTILE = 0.995  # fixed, a-priori alert threshold rule
EWMA_ALPHA = 0.05  # ~ matches the criticality counter's sustained-detection idea


def _ewma(score: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA-smoothed score series (causal; preserves time order)."""
    score = np.asarray(score, dtype=np.float64)
    if len(score) == 0:
        return score
    out = np.empty_like(score)
    out[0] = score[0]
    for i in range(1, len(score)):
        out[i] = alpha * score[i] + (1 - alpha) * out[i - 1]
    return out


def evaluate_farm(farm: str, model_name: str, producing_only_train: bool = True,
                  smooth_alpha: float | None = None,
                  quantile: float = TRAIN_QUANTILE, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    ds_scores: List[DatasetScore] = []
    dataset_level_score = []
    dataset_level_label = []
    per_dataset = []

    for ds in D.iter_care(farm):
        train_df = ds.train_frame(producing_only=producing_only_train)
        pred_df = ds.predict_frame(producing_only=False)
        if len(train_df) < 200 or len(pred_df) < 10:
            continue

        pre = Preprocessor(ds.feature_cols).fit(train_df)
        if pre.n_features == 0:
            continue
        Xtr = pre.transform(train_df)
        Xpr = pre.transform(pred_df)

        scorer = make_scorer(model_name)
        scorer.fit(Xtr)
        s_tr = scorer.score(Xtr)
        s_pr = scorer.score(Xpr)

        # Optional leak-free EWMA smoothing: applied identically to train and
        # prediction scores; threshold still derived from train only.
        if smooth_alpha is not None:
            s_tr = _ewma(s_tr, smooth_alpha)
            s_pr = _ewma(s_pr, smooth_alpha)

        thr = float(np.quantile(s_tr, quantile))

        # Align status + event mask to the prediction frame actually scored.
        pred_index = pred_df.index.to_numpy()
        status = ds.df.loc[pred_index, "status_type_id"].to_numpy()
        event_mask = ds.event_mask[pred_index]

        ds_scores.append(DatasetScore(
            label=ds.label, score=s_pr, status=status,
            event_mask=event_mask, threshold=thr,
        ))

        # Dataset-level summary: peak of the (smoothed-or-raw) prediction score.
        peak_series = s_pr if smooth_alpha is not None else _ewma(s_pr, EWMA_ALPHA)
        dataset_level_score.append(float(np.max(peak_series)))
        dataset_level_label.append(1 if ds.label == "anomaly" else 0)
        per_dataset.append({
            "event_id": ds.event_id, "label": ds.label,
            "n_train": len(train_df), "n_pred": len(pred_df),
            "n_features": pre.n_features,
            "ewma_peak": round(dataset_level_score[-1], 4),
        })
        if verbose:
            print(f"  [{farm}] ev{ds.event_id:>3} {ds.label:<7} "
                  f"feat={pre.n_features:<4} peak={dataset_level_score[-1]:.3f}")

    care = compute_care(ds_scores)

    labels = np.array(dataset_level_label)
    scores = np.array(dataset_level_score)
    event_auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    event_ap = float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else float("nan")

    result = {
        "farm": farm,
        "model": model_name,
        "smooth_alpha": smooth_alpha,
        "n_datasets": len(ds_scores),
        "train_quantile": quantile,
        "care": care,
        "event_level_auc": round(event_auc, 4),
        "event_level_ap": round(event_ap, 4),
        "runtime_sec": round(time.perf_counter() - t0, 1),
        "per_dataset": per_dataset,
    }
    # Attach in-memory artifacts (numpy/objects) for cross-farm aggregation.
    # Not JSON-serialized; consumed by run_all.py.
    result["_ds_scores"] = ds_scores
    result["_level_scores"] = dataset_level_score
    result["_level_labels"] = dataset_level_label
    return result


def _json_safe(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="A", help="comma-separated: A,B,C")
    ap.add_argument("--models", default="PCARecon,Mahalanobis,IsolationForest",
                    help="comma-separated scorer names")
    ap.add_argument("--smooth", type=float, default=None,
                    help="EWMA alpha for score smoothing (e.g. 0.05); omit for raw")
    ap.add_argument("--quantile", type=float, default=TRAIN_QUANTILE,
                    help="train-score quantile for the alert threshold")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    farms = [f.strip() for f in args.farms.split(",") if f.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summary = []
    for model_name in models:
        for farm in farms:
            print(f"\n=== {model_name} on Farm {farm} (smooth={args.smooth}) ===")
            res = evaluate_farm(farm, model_name, smooth_alpha=args.smooth)
            tag = f"_s{args.smooth}" if args.smooth is not None else ""
            out = RESULTS_DIR / f"care_{farm}__{model_name}{tag}.json"
            out.write_text(json.dumps(_json_safe(res), indent=2))
            c = res["care"]
            print(f"  -> CARE={c['care_score']}  event_AUC={res['event_level_auc']}  "
                  f"AP={res['event_level_ap']}  "
                  f"(cov={c['coverage_fbeta']} ws={c['earliness_ws']} "
                  f"ef={c['reliability_efbeta']} acc={c['accuracy']}; "
                  f"tp={c['event_tp']} fp={c['event_fp']} fn={c['event_fn']})")
            summary.append({
                "model": model_name, "farm": farm,
                "care_score": c["care_score"],
                "event_auc": res["event_level_auc"],
                "event_ap": res["event_level_ap"],
            })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(pd.DataFrame(summary).to_string(index=False))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
