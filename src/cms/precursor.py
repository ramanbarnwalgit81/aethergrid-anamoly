"""
CARE-Precursor: a predictive re-evaluation of the CARE benchmark.

The original CARE prediction window begins essentially at the operator's logged
event_start, so a model is graded on detecting a fault the operator has *already*
flagged — and the scoreable window leaves almost no room to fire earlier
(Sec. 7 of the paper: median lead time is negative). We define the genuinely
*predictive* task instead:

  precursor window = [event_start - LEAD, event_start), restricted to NORMAL
                     operating status (the turbine still looks healthy);
  training        = only EARLIER healthy history (id < event_start - LEAD - GUARD),
                     so the precursor is never seen during training;
  dataset score   = aggregate anomaly score over the precursor window;
  label           = anomaly (heading to failure) vs normal (reference) dataset.

The question: from healthy-looking data BEFORE the operator notices, can we tell
which turbines are heading to failure? This is the deployable question, and we
release the precursor windows so others can target it. We report it honestly with
bootstrap CIs; we expect it to be hard (near-chance), which is the point — it is
the unsolved task the field should be evaluated on.

Outputs: docs/results/cms/precursor_<farms>.json and released window specs in
data/benchmark/care_precursor_v2/. Real runs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.cms import data as D
from src.cms.models import make_scorer
from src.cms.preprocess import Preprocessor

RESULTS_DIR = Path("docs/results/cms")
SPEC_DIR = Path("data/benchmark/care_precursor_v2")
NORMAL_STATUS = {0, 1, 2}
LEAD = 1008    # 1 week at 10-min resolution
GUARD = 144    # 1-day buffer between training history and the precursor window


def _ewma(s, alpha=0.05):
    s = np.asarray(s, float)
    if len(s) == 0:
        return s
    out = np.empty_like(s)
    out[0] = s[0]
    for i in range(1, len(s)):
        out[i] = alpha * s[i] + (1 - alpha) * out[i - 1]
    return out


def run(farms, model_name="Mahalanobis", lead=LEAD, guard=GUARD, agg="mean"):
    ds_scores, labels, specs, per_ds = [], [], [], []
    for farm in farms:
        for ds in D.iter_care(farm):
            es = ds.event_start_id
            ids = ds.df["id"].to_numpy()
            status = ds.df["status_type_id"].to_numpy()
            normal = np.isin(status, list(NORMAL_STATUS))

            prec_mask = (ids >= es - lead) & (ids < es) & normal
            train_mask = (ids < es - lead - guard) & normal
            if train_mask.sum() < 500 or prec_mask.sum() < 30:
                continue

            train_df = ds.df.loc[train_mask]
            prec_df = ds.df.loc[prec_mask]
            pre = Preprocessor(ds.feature_cols).fit(train_df)
            if pre.n_features == 0:
                continue
            sc = make_scorer(model_name)
            sc.fit(pre.transform(train_df))
            s = _ewma(sc.score(pre.transform(prec_df)))
            score = float(np.mean(s)) if agg == "mean" else float(np.max(s))

            ds_scores.append(score)
            labels.append(1 if ds.label == "anomaly" else 0)
            specs.append({"farm": farm, "event_id": ds.event_id, "label": ds.label,
                          "precursor_start_id": int(es - lead),
                          "precursor_end_id": int(es),
                          "n_precursor_rows": int(prec_mask.sum())})
            per_ds.append({"farm": farm, "event_id": ds.event_id,
                           "label": ds.label, "precursor_score": round(score, 5)})

    labels = np.array(labels); scores = np.array(ds_scores)
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    ap = float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    # bootstrap CI on AUC
    rng = np.random.default_rng(42); boots = []
    for _ in range(3000):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx])) > 1:
            boots.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))

    result = {
        "task": "CARE-Precursor (predictive: detect before operator event_start)",
        "model": model_name, "farms": farms, "lead_steps": lead, "guard_steps": guard,
        "lead_hours": lead * 10 / 60, "aggregation": agg,
        "n_datasets": int(len(labels)),
        "n_anomaly": int(labels.sum()), "n_normal": int((labels == 0).sum()),
        "precursor_auc": round(auc, 4),
        "precursor_auc_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "precursor_ap": round(ap, 4),
        "per_dataset": per_ds,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"precursor_{''.join(farms)}_{model_name}.json").write_text(
        json.dumps(result, indent=2))
    # Release the precursor window specification (benchmark artifact).
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    (SPEC_DIR / f"precursor_windows_{''.join(farms)}.json").write_text(
        json.dumps({"lead_steps": lead, "guard_steps": guard, "windows": specs}, indent=2))

    print(f"\nCARE-Precursor ({model_name}) farms {','.join(farms)} | "
          f"lead {result['lead_hours']:.0f} h")
    print(f"  datasets: {result['n_datasets']} ({result['n_anomaly']} anomaly / "
          f"{result['n_normal']} normal)")
    print(f"  PRECURSOR AUC = {result['precursor_auc']} "
          f"CI95 {result['precursor_auc_ci95']}  AP = {result['precursor_ap']}")
    print(f"  (vs in-window detection AUC ~0.66; chance = 0.50)")
    print(f"[OK] wrote precursor results + released window spec to {SPEC_DIR}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="A,B")
    ap.add_argument("--model", default="Mahalanobis")
    ap.add_argument("--lead", type=int, default=LEAD)
    ap.add_argument("--agg", default="mean", choices=["mean", "peak"])
    args = ap.parse_args()
    run([f.strip() for f in args.farms.split(",")], args.model, args.lead, agg=args.agg)


if __name__ == "__main__":
    main()
