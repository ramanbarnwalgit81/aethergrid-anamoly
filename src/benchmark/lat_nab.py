"""Run the Label Ambiguity Test (LAT) on the Numenta Anomaly Benchmark.

NAB ships with two native label conventions for the same set of series:
  - combined_labels.json   : point anomalies (exact timestamps)
  - combined_windows.json  : operator-smoothed anomaly windows around the
                              points with tolerance on either side

This is the same "point-alert vs operational-window" ambiguity the paper
addresses on CARE (`event_start_id` vs `status_type_id != 0`) and a natural
second-benchmark test of the claim that LAT generalises beyond CARE.

We download the two label JSONs and the raw time-series CSVs from the NAB
GitHub mirror, compute LAS_data on every series, and compute the full
LAS_{data,model,rank} triple on a diverse sample that we can afford to
score with three unsupervised detectors (Isolation Forest, PCA reconstruction
error, rolling MAD) on a CPU.

Output:  docs/results/lat_nab.json

Usage:  python -m src.benchmark.lat_nab
"""
from __future__ import annotations
from pathlib import Path
import json
import urllib.request
import io

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau

RESULTS_DIR = Path("docs/results")
NAB_RAW = "https://raw.githubusercontent.com/numenta/NAB/master"
# Diverse subset of real-world NAB series (excludes purely synthetic /
# artificialWithAnomaly / artificialNoAnomaly because those are noise-free by
# construction and do not stress LAT).
SUBSET = [
    "realKnownCause/nyc_taxi.csv",
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "realKnownCause/ec2_request_latency_system_failure.csv",
    "realKnownCause/machine_temperature_system_failure.csv",
    "realKnownCause/rogue_agent_key_hold.csv",
    "realTraffic/occupancy_6005.csv",
    "realTraffic/speed_7578.csv",
    "realTraffic/TravelTime_387.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
]


def fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def windows_to_mask(windows: list, timestamps: pd.Series) -> np.ndarray:
    """Convert a list of [start_iso, end_iso] windows to a 0/1 label array."""
    mask = np.zeros(len(timestamps), dtype=int)
    for start, end in windows:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        mask |= ((timestamps >= s) & (timestamps <= e)).to_numpy().astype(int)
    return mask


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(((a == 1) & (b == 1)).sum())
    union = int(((a == 1) | (b == 1)).sum())
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def score_rolling_mad(values: np.ndarray, window: int = 100) -> np.ndarray:
    s = pd.Series(values)
    median = s.rolling(window, min_periods=1).median()
    mad = (s - median).abs().rolling(window, min_periods=1).median() * 1.4826 + 1e-6
    return (np.abs(s - median) / mad).to_numpy()


def score_isolation_forest(values: np.ndarray, seed: int = 0) -> np.ndarray:
    X = values.reshape(-1, 1)
    model = IsolationForest(n_estimators=100, random_state=seed, contamination="auto")
    model.fit(X)
    return -model.score_samples(X)


def score_pca(values: np.ndarray, k: int = 1) -> np.ndarray:
    # single-variable NAB series: PCA on lag-embedded features
    L = 10
    n = len(values)
    X = np.zeros((n, L))
    for i in range(L):
        X[i:, i] = values[: n - i]
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k).fit(Xs)
    recon = pca.inverse_transform(pca.transform(Xs))
    return np.linalg.norm(Xs - recon, axis=1)


def las_for_series(df: pd.DataFrame, y_tight: np.ndarray, y_relax: np.ndarray,
                    compute_full: bool = True) -> dict:
    """Compute LAS triple on one series. Returns dict with LAS_data and (if
    compute_full) LAS_model, LAS_rank."""
    out = {"las_data_jaccard": round(jaccard_distance(y_tight, y_relax), 4)}
    if not compute_full:
        return out
    if len(np.unique(y_tight)) < 2 or len(np.unique(y_relax)) < 2:
        out["las_model_mean_delta_auc"] = None
        out["las_rank_one_minus_tau"] = None
        return out

    values = df.iloc[:, 1].astype(float).to_numpy()
    scorers = {
        "iforest_s0": lambda v: score_isolation_forest(v, seed=0),
        "iforest_s1": lambda v: score_isolation_forest(v, seed=1),
        "pca":        lambda v: score_pca(v),
        "rolling_mad": lambda v: score_rolling_mad(v),
    }
    deltas = []
    tau_deltas = []
    for _, fn in scorers.items():
        try:
            s = fn(values)
        except Exception:
            continue
        a_tight = roc_auc_score(y_tight, s)
        a_relax = roc_auc_score(y_relax, s)
        deltas.append(abs(a_tight - a_relax))

        # LAS_rank: Kendall tau between scores on union-of-positives
        union = (y_tight | y_relax).astype(bool)
        if union.sum() >= 3:
            s_union = s[union]
            # rank under tight-label sort vs relax-label sort on positive indices
            t, _ = kendalltau(y_tight[union] * 1000 + s_union,
                                y_relax[union] * 1000 + s_union)
            if t is not None and not np.isnan(t):
                tau_deltas.append(1.0 - t)

    out["las_model_mean_delta_auc"] = round(float(np.mean(deltas)), 4) if deltas else None
    out["las_rank_one_minus_tau"] = round(float(np.mean(tau_deltas)), 4) if tau_deltas else None
    return out


def main():
    print("[1/4] Downloading NAB label files...")
    windows_url = f"{NAB_RAW}/labels/combined_windows.json"
    points_url  = f"{NAB_RAW}/labels/combined_labels.json"
    tight_json = json.loads(fetch(points_url))   # point labels → "tight"
    relax_json = json.loads(fetch(windows_url))  # windows     → "lenient"
    # points are lists of ISO timestamps; normalise to windows of length 0
    tight = {k: [[ts, ts] for ts in v] for k, v in tight_json.items()}
    relax = relax_json
    print(f"  points (tight): {len(tight)} series, windows (lenient): {len(relax)} series")

    # LAS_data for every series
    print("[2/4] Computing LAS_data for all 58 NAB series…")
    las_data_all = []
    for series_path in tight:
        if series_path not in relax:
            continue
        tight_win = tight[series_path]
        relax_win = relax[series_path]
        if not tight_win and not relax_win:
            continue  # no anomalies under either convention
        # For LAS_data without the CSV we use a synthetic dense timeline:
        # approximate by treating window-ENDPOINT sets rather than rows. A
        # full row-level Jaccard requires the CSV timestamps; we'll compute
        # that for the full-LAT subset.
        # Here we report endpoint-set Jaccard as a quick screen.
        t_eps = {w[0] for w in tight_win} | {w[1] for w in tight_win}
        r_eps = {w[0] for w in relax_win} | {w[1] for w in relax_win}
        j = 1 - len(t_eps & r_eps) / max(1, len(t_eps | r_eps))
        las_data_all.append({"series": series_path, "endpoint_jaccard": round(j, 4)})
    all_j = [r["endpoint_jaccard"] for r in las_data_all]
    print(f"  endpoint-Jaccard across {len(all_j)} labelled series: "
          f"mean={np.mean(all_j):.3f}, median={np.median(all_j):.3f}")

    # Full LAT on the subset
    print("[3/4] Running full LAT on subset with detectors…")
    subset_results = []
    for series_path in SUBSET:
        if series_path not in tight or series_path not in relax:
            print(f"  [skip] {series_path} not in both label files")
            continue
        csv_url = f"{NAB_RAW}/data/{series_path}"
        try:
            raw = fetch(csv_url)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            print(f"  [skip] {series_path}: {e}")
            continue
        df.columns = ["timestamp", "value"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        y_tight = windows_to_mask(tight[series_path], df["timestamp"])
        y_relax = windows_to_mask(relax[series_path], df["timestamp"])
        r = las_for_series(df, y_tight, y_relax, compute_full=True)
        r["series"] = series_path
        r["n_rows"] = int(len(df))
        r["n_tight"] = int(y_tight.sum())
        r["n_relax"] = int(y_relax.sum())
        subset_results.append(r)
        print(f"  {series_path:<60s}  "
              f"LAS_data={r['las_data_jaccard']:.3f}  "
              f"LAS_model={r['las_model_mean_delta_auc']}  "
              f"LAS_rank={r['las_rank_one_minus_tau']}")

    # Aggregate
    print("[4/4] Aggregating…")
    d = [r["las_data_jaccard"] for r in subset_results if r["las_data_jaccard"] is not None]
    m = [r["las_model_mean_delta_auc"] for r in subset_results
          if r["las_model_mean_delta_auc"] is not None]
    k = [r["las_rank_one_minus_tau"] for r in subset_results
          if r["las_rank_one_minus_tau"] is not None]
    aggregate = {
        "n_series_subset": len(subset_results),
        "mean_las_data":  round(float(np.mean(d)), 4) if d else None,
        "mean_las_model": round(float(np.mean(m)), 4) if m else None,
        "mean_las_rank":  round(float(np.mean(k)), 4) if k else None,
    }
    print(f"\nNAB aggregate (n={len(subset_results)} series):")
    print(f"  mean LAS_data  = {aggregate['mean_las_data']}")
    print(f"  mean LAS_model = {aggregate['mean_las_model']}")
    print(f"  mean LAS_rank  = {aggregate['mean_las_rank']}")
    print()
    print(f"CARE Farm A reference (from paper, v6): LAS_data=0.778, "
          f"LAS_model=0.082, LAS_rank=0.311")

    out = {
        "benchmark": "Numenta Anomaly Benchmark (NAB)",
        "label_conventions": ["combined_windows.json", "combined_windows_relaxed.json"],
        "endpoint_jaccard_all_series": las_data_all,
        "subset_full_lat": subset_results,
        "aggregate_subset": aggregate,
        "care_farm_a_reference": {
            "las_data": 0.778, "las_model": 0.082, "las_rank": 0.311,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "lat_nab.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[OK] Saved: docs/results/lat_nab.json")


if __name__ == "__main__":
    main()
