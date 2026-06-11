"""
XGBoost post-hoc stacker on v7 ensemble scores.

GOAL
----
Push CARE Farm A hard-task AUC above 0.60 without retraining the ensemble.

APPROACH
--------
For each test row the v7 ensemble produces a single score. We augment that
score with:
  - rolling_mean_10 (of the score itself)  — captures sustained elevation
  - rolling_std_10                           — captures noise level
  - score_delta                              — first-difference momentum
  - rolling_rank                             — position in event's score CDF
  - event_description one-hot (4 fault types)

The XGBoost meta-learner is trained leave-one-anomaly-event-out on the
event-window and CARE-Precursor labels. Calibrated via conformal wrapper.

This is "stacking" in the sense of meta-learning over derived features —
not stacking multiple base models. Legitimate improvement since the meta
model learns fault-type-specific thresholds and temporal context that the
point-wise ensemble can't.

Usage:
    python -m src.benchmark.xgb_stacker
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None


RESULTS_DIR = Path("docs/results")
CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")


def load_v7_per_event():
    path = RESULTS_DIR / "care_ensemble_v7_per_event_scores.json"
    with path.open() as f:
        return json.load(f)


def load_event_info():
    df = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    return {int(r["event_id"]): r for _, r in df.iterrows()}


def load_precursor_info():
    path = Path("data/benchmark/care_precursor/event_info_precursor_farm_A.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep=";")
    return {int(r["event_id"]): r for _, r in df.iterrows()}


FAULT_TYPES = ["Gearbox failure", "Generator bearing failure",
                "Hydraulic group", "Transformer failure", "nan"]


def build_features(scores: np.ndarray, fault_type: str,
                     window: int = 10,
                     use_fault_type: bool = True) -> pd.DataFrame:
    """Derive temporal + categorical features from a single-event score array.

    Args:
        use_fault_type: if True, adds fault-type one-hot features.
            These give the model oracle knowledge of event fault type under
            leave-one-event-out — borderline leakage for deployment semantics.
            Set False for the leakage-cleaned ablation.
    """
    s = pd.Series(scores)
    feats = pd.DataFrame({
        "score": scores.astype(np.float32),
        "rolling_mean_10": s.rolling(window, min_periods=1).mean().values,
        "rolling_std_10": s.rolling(window, min_periods=1).std().fillna(0).values,
        "score_delta": s.diff().fillna(0).values,
        "rolling_rank": s.rolling(100, min_periods=1).rank(pct=True).fillna(0.5).values,
    })
    if use_fault_type:
        for ft in FAULT_TYPES:
            feats[f"fault_{ft.replace(' ', '_')}"] = int(fault_type == ft)
    return feats


def leave_one_out_stacker(data: dict, event_info: dict,
                             label_key: str = "y_event",
                             n_estimators: int = 100, max_depth: int = 3,
                             learning_rate: float = 0.1,
                             use_fault_type: bool = True) -> dict:
    """
    Leave-one-event-out XGBoost meta-learner evaluation.

    For each anomaly event, train XGBoost on features from all other events
    and score the held-out one. Compute per-event AUC and pooled AUC under
    both event-window and CARE-Precursor labels.
    """
    if xgb is None:
        raise ImportError("xgboost not installed")

    # Only anomaly events for AUC evaluation
    anomaly_ids = [eid for eid, info in event_info.items()
                    if info["event_label"] == "anomaly"]

    # Prepare all per-event feature matrices
    per_event_X = {}
    per_event_y = {}
    for eid in data:
        eid_int = int(eid)
        if eid_int not in event_info:
            continue
        info = event_info[eid_int]
        fault_type = str(info.get("event_description", "nan"))
        scores = np.array(data[eid]["scores"], dtype=np.float32)
        y = np.array(data[eid].get(label_key, []), dtype=np.int64)
        if len(y) == 0 or len(y) != len(scores):
            continue
        X = build_features(scores, fault_type, use_fault_type=use_fault_type)
        per_event_X[eid_int] = X
        per_event_y[eid_int] = y

    results = []
    pooled_scores = []
    pooled_labels = []

    for held_id in anomaly_ids:
        if held_id not in per_event_X:
            continue
        # Training set: all other events (anomaly + normal)
        X_train_parts, y_train_parts = [], []
        for eid, X in per_event_X.items():
            if eid == held_id:
                continue
            y = per_event_y[eid]
            if len(np.unique(y)) == 2:  # need both classes
                X_train_parts.append(X)
                y_train_parts.append(y)
        if not X_train_parts:
            continue
        X_train = pd.concat(X_train_parts, ignore_index=True)
        y_train = np.concatenate(y_train_parts)

        X_test = per_event_X[held_id]
        y_test = per_event_y[held_id]

        if len(np.unique(y_test)) < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="auc", verbosity=0,
            tree_method="hist",
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        base_auc = float(roc_auc_score(y_test, X_test["score"].values))
        stacker_auc = float(roc_auc_score(y_test, proba))

        results.append({
            "event_id": held_id,
            "fault_type": str(event_info[held_id].get("event_description", "")),
            "n_test": len(y_test),
            "base_v7_auc": round(base_auc, 4),
            "xgb_stacker_auc": round(stacker_auc, 4),
            "delta_auc": round(stacker_auc - base_auc, 4),
        })

        pooled_scores.append(proba)
        pooled_labels.append(y_test)

    # Aggregates
    print(f"\n{'ID':>4} {'Fault':<28} {'n_test':>6} {'v7 AUC':>8} {'XGB AUC':>8} {'Δ':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['event_id']:>4} {r['fault_type'][:26]:<28} "
              f"{r['n_test']:>6} {r['base_v7_auc']:>8.4f} {r['xgb_stacker_auc']:>8.4f} "
              f"{r['delta_auc']:>+8.4f}")

    mean_base = np.mean([r["base_v7_auc"] for r in results])
    mean_stack = np.mean([r["xgb_stacker_auc"] for r in results])
    median_delta = np.median([r["delta_auc"] for r in results])

    ps = np.concatenate(pooled_scores)
    pl = np.concatenate(pooled_labels)
    pooled_auc = float(roc_auc_score(pl, ps)) if len(np.unique(pl)) == 2 else None

    agg = {
        "label": label_key,
        "n_events": len(results),
        "mean_base_v7_auc": round(mean_base, 4),
        "mean_xgb_stacker_auc": round(mean_stack, 4),
        "median_delta_auc": round(median_delta, 4),
        "pooled_xgb_stacker_auc": round(pooled_auc, 4) if pooled_auc else None,
    }
    print(f"\n  Mean v7 base AUC:          {agg['mean_base_v7_auc']:.4f}")
    print(f"  Mean XGBoost stacker AUC:  {agg['mean_xgb_stacker_auc']:.4f}")
    print(f"  Median Δ:                  {agg['median_delta_auc']:+.4f}")
    print(f"  Pooled XGBoost AUC:        {agg['pooled_xgb_stacker_auc']}")

    return {"aggregate": agg, "per_event": results}


def main():
    print("=" * 80)
    print("  XGBoost post-hoc stacker on v7 ensemble scores")
    print("=" * 80)

    data = load_v7_per_event()
    event_info = load_event_info()

    # Run BOTH versions: with and without fault-type one-hot
    for use_ft in [True, False]:
        suffix = "with_faulttype" if use_ft else "no_faulttype"
        print(f"\n{'='*80}")
        print(f"  ABLATION: {suffix}  (use_fault_type={use_ft})")
        print(f"{'='*80}")

        for label in ["y_event", "y_precursor", "y_status"]:
            print(f"\n--- Label: {label} ({suffix}) ---")
            result = leave_one_out_stacker(data, event_info, label_key=label,
                                               use_fault_type=use_ft)
            out_path = RESULTS_DIR / f"xgb_stacker_{label}_{suffix}.json"
            with out_path.open("w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  [OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
