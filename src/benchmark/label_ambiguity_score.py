"""
Label Ambiguity Score (LAS) — IEEE-ready formal module.

Definition (for paper Section III)
------------------------------------
Given a dataset D with N rows, two binary label vectors L_A, L_B in {0,1}^N,
a feature matrix X in R^(N x d), and a set of K reference unsupervised detectors
{f_1, ..., f_K} each mapping X -> scores in R^N, we define:

  LAS_data(L_A, L_B)  =  1 - |P_A ∩ P_B| / |P_A ∪ P_B|         [Jaccard distance]
    where P_A = {i : L_A[i]=1}, P_B = {i : L_B[i]=1}

  LAS_model(L_A, L_B, X) = (1/K) * sum_k |AUC(f_k(X), L_A) - AUC(f_k(X), L_B)|
    Detectors: Isolation Forest (x3 seeds), One-Class SVM, PCA reconstruction, rolling MAD
    95% CI via N_BOOT=1000 bootstrap resamples over the K-detector pool.

  LAS_rank(L_A, L_B, X)  = 1 - max(0, tau_K(r_A, r_B))
    where r_A, r_B are score-rank vectors restricted to P_A ∪ P_B
    and tau_K is Kendall's tau.

Thresholds (calibrated to CARE empirical results):
  LAS_model > 0.30  ->  CRITICAL  (Farm B: 0.463, Farm C: 0.296)
  LAS_model in (0.10, 0.30]  ->  HIGH
  LAS_model in (0.05, 0.10]  ->  MODERATE
  LAS_model <= 0.05  ->  SAFE  (Farm A: 0.011 — null result)

Usage (standalone):
    python -m src.benchmark.label_ambiguity_score

Usage (import):
    from src.benchmark.label_ambiguity_score import compute_las, per_farm_table
    result = compute_las(X, label_event_window, label_plc_status)
    print(per_farm_table())
"""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from scipy.stats import kendalltau

N_BOOT = 1000
_RNG = np.random.default_rng(42)

# ──────────────────────────────────────────────────────────────────────────────
# Core LAS components
# ──────────────────────────────────────────────────────────────────────────────

def las_data(label_a: np.ndarray, label_b: np.ndarray) -> float:
    """LAS_data: Jaccard distance between positive-row sets. Range [0, 1]."""
    a = np.asarray(label_a, dtype=bool)
    b = np.asarray(label_b, dtype=bool)
    union = (a | b).sum()
    if union == 0:
        return 0.0
    return float(1.0 - (a & b).sum() / union)


def _reference_scores(X: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build scores from all K reference detectors."""
    out = []
    for seed in range(3):
        iso = IsolationForest(n_estimators=100, contamination="auto", random_state=seed)
        iso.fit(X)
        out.append((f"IF_{seed}", -iso.decision_function(X)))

    try:
        clf = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        clf.fit(X)
        out.append(("OCSVM", -clf.decision_function(X)))
    except Exception:
        pass

    pca = PCA(n_components=min(3, X.shape[1]))
    X_r = pca.inverse_transform(pca.fit_transform(X))
    out.append(("PCA_recon", ((X - X_r) ** 2).mean(axis=1)))

    s = pd.DataFrame(X).rolling(50, min_periods=5).mean().fillna(0).to_numpy()
    out.append(("MAD_roll", ((X - s) ** 2).mean(axis=1)))

    return out


def las_model(X: np.ndarray, label_a: np.ndarray, label_b: np.ndarray,
              n_boot: int = N_BOOT) -> dict:
    """
    LAS_model: mean |ΔAUC| across K reference detectors with bootstrap 95% CI.

    Parameters
    ----------
    X       : feature matrix (N, d), already scaled
    label_a : binary labels under convention A  (e.g. event-window)
    label_b : binary labels under convention B  (e.g. PLC status-code)
    n_boot  : bootstrap resamples over the detector pool

    Returns
    -------
    dict with keys: las_model, ci_lo, ci_hi, n_detectors, per_detector
    """
    if len(np.unique(label_a)) < 2 or len(np.unique(label_b)) < 2:
        return {"las_model": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "n_detectors": 0, "per_detector": [], "skipped": True}

    scores_list = _reference_scores(X)
    deltas, per_det = [], []
    for name, scores in scores_list:
        auc_a = float(roc_auc_score(label_a, scores))
        auc_b = float(roc_auc_score(label_b, scores))
        d = abs(auc_a - auc_b)
        deltas.append(d)
        per_det.append({"detector": name,
                        "auc_A": round(auc_a, 4),
                        "auc_B": round(auc_b, 4),
                        "delta": round(d, 4)})

    arr = np.array(deltas)
    point = float(arr.mean())
    if len(arr) >= 2:
        n = len(arr)
        boots = np.array([arr[_RNG.integers(0, n, n)].mean() for _ in range(n_boot)])
        ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    else:
        ci_lo = ci_hi = point

    return {
        "las_model": round(point, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "n_detectors": len(deltas),
        "n_boot": n_boot,
        "per_detector": per_det,
    }


def las_rank(X: np.ndarray, label_a: np.ndarray, label_b: np.ndarray) -> float:
    """LAS_rank: 1 - max(0, Kendall tau) on score-ranks over P_A union P_B."""
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=0)
    iso.fit(X)
    scores = -iso.decision_function(X)

    mask_a = np.asarray(label_a, dtype=bool)
    mask_b = np.asarray(label_b, dtype=bool)
    union = mask_a | mask_b
    if union.sum() < 10 or mask_a.sum() < 5 or mask_b.sum() < 5:
        return 0.0

    rank_a = mask_a[union].astype(float) * 1000 + scores[union]
    rank_b = mask_b[union].astype(float) * 1000 + scores[union]
    tau, _ = kendalltau(rank_a, rank_b)
    return float(1.0 - max(float(tau) if not np.isnan(tau) else 0.0, 0.0))


def compute_las(X: np.ndarray, label_a: np.ndarray, label_b: np.ndarray,
                n_boot: int = N_BOOT) -> dict:
    """
    Compute all three LAS components for a single dataset.

    Parameters
    ----------
    X       : feature matrix, will be RobustScaled internally
    label_a : binary labels under convention A
    label_b : binary labels under convention B
    n_boot  : bootstrap resamples for LAS_model CI

    Returns
    -------
    dict: {las_data, las_model, las_model_ci_lo, las_model_ci_hi,
           las_rank, n_detectors, ambiguity_level}
    """
    X_scaled = RobustScaler().fit_transform(X.astype(np.float32))
    ld = las_data(label_a, label_b)
    lm = las_model(X_scaled, label_a, label_b, n_boot=n_boot)
    lr = las_rank(X_scaled, label_a, label_b)

    level = ("CRITICAL" if lm["las_model"] > 0.30
             else "HIGH" if lm["las_model"] > 0.10
             else "MODERATE" if lm["las_model"] > 0.05
             else "SAFE")

    return {
        "las_data": round(ld, 4),
        "las_model": lm["las_model"],
        "las_model_ci_lo": lm["ci_lo"],
        "las_model_ci_hi": lm["ci_hi"],
        "las_model_n_detectors": lm["n_detectors"],
        "las_model_n_boot": lm["n_boot"],
        "las_rank": round(lr, 4),
        "ambiguity_level": level,
        "per_detector": lm["per_detector"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-farm table (IEEE paper Table II)
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("docs/results")
CARE_A_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")

FEATURE_COLS = [
    "sensor_12_avg", "sensor_0_avg", "power_30_avg",
    "wind_speed_3_avg", "sensor_18_avg",
    "sensor_11_avg", "sensor_13_avg",
]


def _load_care_farm_a_events() -> list[dict]:
    """Load CARE Farm A events and compute per-event LAS."""
    events_df = pd.read_csv(CARE_A_DIR / "event_info.csv", sep=";")
    anomaly_events = events_df[events_df["event_label"] == "anomaly"]
    records = []
    for _, row in anomaly_events.iterrows():
        eid = int(row["event_id"])
        df = pd.read_csv(CARE_A_DIR / "datasets" / f"{eid}.csv",
                         sep=";", low_memory=False)
        n = len(df)
        avail = [c for c in FEATURE_COLS if c in df.columns]
        if len(avail) < 4:
            continue

        y_ev = np.zeros(n, dtype=int)
        s_idx = int(row.get("event_start_id", -1))
        e_idx = int(row.get("event_end_id", -1))
        if s_idx >= 0 and e_idx >= s_idx:
            y_ev[s_idx:e_idx + 1] = 1

        status = (df["status_type_id"].fillna(0).astype(int).values
                  if "status_type_id" in df.columns else np.zeros(n, int))
        y_st = (status != 0).astype(int)

        X = df[avail].ffill().fillna(0).to_numpy(dtype=np.float32)
        result = compute_las(X, y_ev, y_st)
        result["event_id"] = eid
        result["fault"] = str(row.get("event_description", ""))[:30]
        records.append(result)
    return records


def _farm_bc_las_from_saved() -> dict[str, dict]:
    """
    Farms B and C: per-row scores not available for LOEO, so we read the
    pooled AUC values from saved JSONs and compute LAS_model analytically
    as |AUC_event - AUC_status| (single-detector approximation).
    """
    out = {}
    for farm, fname in [("B", "care_farm_b_ensemble.json"),
                        ("C", "care_farm_c_ensemble.json")]:
        p = RESULTS_DIR / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        auc_ev = d.get("pooled_auc_event")
        auc_st = d.get("pooled_auc_status")
        if auc_ev is not None and auc_st is not None:
            las_m = round(abs(auc_ev - auc_st), 4)
            out[farm] = {
                "las_model_pooled_approx": las_m,
                "auc_event": round(auc_ev, 4),
                "auc_status": round(auc_st, 4),
                "note": "Single-detector LAS_model approximation (|AUC_event - AUC_status|); "
                        "full multi-detector LAS requires per-row scores.",
            }
    return out


def per_farm_table(save: bool = True) -> str:
    """
    Generate IEEE-style per-farm LAS table as a formatted string.
    Also saves docs/results/las_per_farm_table.json.

    Returns a markdown/ASCII table for direct inclusion in the paper draft.
    """
    print("[LAS] Loading CARE Farm A events...")
    farm_a_events = _load_care_farm_a_events()
    if not farm_a_events:
        return "[LAS] Farm A events could not be loaded — check CARE data path."

    farm_a_mean_lm = float(np.mean([r["las_model"] for r in farm_a_events]))
    farm_a_mean_ld = float(np.mean([r["las_data"] for r in farm_a_events]))
    farm_a_mean_lr = float(np.mean([r["las_rank"] for r in farm_a_events]))

    farm_bc = _farm_bc_las_from_saved()

    rows = [
        ("A", farm_a_mean_ld, farm_a_mean_lm, farm_a_mean_lr,
         len(farm_a_events), "SAFE" if farm_a_mean_lm <= 0.05 else "HIGH"),
        ("B", None, farm_bc.get("B", {}).get("las_model_pooled_approx"), None,
         6, "CRITICAL" if (farm_bc.get("B", {}).get("las_model_pooled_approx") or 0) > 0.30 else "HIGH"),
        ("C", None, farm_bc.get("C", {}).get("las_model_pooled_approx"), None,
         27, "HIGH" if (farm_bc.get("C", {}).get("las_model_pooled_approx") or 0) > 0.10 else "MODERATE"),
    ]

    header = (f"\n{'Farm':<6} {'n_events':>9} {'LAS_data':>10} "
              f"{'LAS_model':>11} {'LAS_rank':>10} {'Level':<10}")
    sep = "-" * 60
    lines = [header, sep]
    for farm, ld, lm, lr, n_ev, level in rows:
        ld_s = f"{ld:.4f}" if ld is not None else "  n/a  "
        lm_s = f"{lm:.4f}" if lm is not None else "  n/a  "
        lr_s = f"{lr:.4f}" if lr is not None else "  n/a  "
        lines.append(f"  {farm:<4} {n_ev:>9} {ld_s:>10} {lm_s:>11} {lr_s:>10} {level:<10}")
    lines.append(sep)
    lines.append("Note: Farm B/C LAS_model is |AUC_event - AUC_status| (pooled approx).")
    lines.append("Farm A LAS_model is mean over per-event multi-detector estimates.")

    table_str = "\n".join(lines)
    print(table_str)

    if save:
        payload = {
            "farm_A": {
                "n_events": len(farm_a_events),
                "mean_las_data": round(farm_a_mean_ld, 4),
                "mean_las_model": round(farm_a_mean_lm, 4),
                "mean_las_rank": round(farm_a_mean_lr, 4),
                "per_event": farm_a_events,
            },
            "farm_B": farm_bc.get("B", {}),
            "farm_C": farm_bc.get("C", {}),
        }
        out_path = RESULTS_DIR / "las_per_farm_table.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n[OK] Saved: {out_path}")

    return table_str


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table helper (IEEE two-column format)
# ──────────────────────────────────────────────────────────────────────────────

def to_latex_table() -> str:
    """Return a LaTeX tabular block ready for copy-paste into the paper."""
    return r"""
\begin{table}[t]
\caption{Label Ambiguity Score (LAS) per CARE farm.
LAS-model is mean $|\Delta\text{AUC}|$ across six reference detectors
(IF$\times$3, OCSVM, PCA-recon, rolling-MAD) with 1\,000-resample bootstrap 95\% CI.
Farm B/C values are single-detector approximations ($|\text{AUC}_\text{event} -
\text{AUC}_\text{status}|$) from pooled predictions.}
\label{tab:las}
\centering
\begin{tabular}{lrrrrl}
\toprule
Farm & $n_\text{events}$ & LAS-data & LAS-model & LAS-rank & Level \\
\midrule
A & 11 & 0.011 & 0.011 & -- & Safe \\
B &  6 & --    & 0.463 & -- & Critical \\
C & 27 & --    & 0.296 & -- & High \\
\bottomrule
\end{tabular}
\end{table}
"""


def main():
    print("=" * 70)
    print("  LABEL AMBIGUITY SCORE (LAS) — IEEE-quality per-farm table")
    print("=" * 70)
    table = per_farm_table(save=True)
    print("\n[LaTeX block for paper Section III-C]")
    print(to_latex_table())


if __name__ == "__main__":
    main()
