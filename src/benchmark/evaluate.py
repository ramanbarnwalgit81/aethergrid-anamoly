"""
WindBench Evaluation Orchestrator.

Runs standardized evaluation of any model against any task from the benchmark.
Every experiment produces a JSON row with identical schema → directly
pushable to the leaderboard.

Usage:
    from src.benchmark.evaluate import run_task
    result = run_task(task_id="T1_detection_care_a",
                       model_name="IsolationForest",
                       scorer_fn=my_score_function)
"""

import json
from pathlib import Path
from dataclasses import asdict
import hashlib

import numpy as np
import pandas as pd

from src.benchmark.registry import REGISTRY, get_manifest
from src.benchmark.tasks import TASKS, TaskType, get_task
from src.benchmark import metrics as M

RESULTS_DIR = Path("data/benchmark/results")


def load_harmonized(dataset_id: str) -> tuple:
    """Load canonical parquet + faults + turbine metadata."""
    m = get_manifest(dataset_id)
    parquet = Path(m.canonical_parquet_path)
    if not parquet.exists():
        raise FileNotFoundError(f"Canonical parquet missing: {parquet}. "
                                 f"Run the adapter for {dataset_id} first.")

    df = pd.read_parquet(parquet)

    faults_path = Path(m.fault_events_csv)
    faults = pd.read_csv(faults_path) if faults_path.exists() else pd.DataFrame()

    turbines_path = Path(m.turbine_metadata_csv)
    turbines = pd.read_csv(turbines_path) if turbines_path.exists() else pd.DataFrame()

    return df, faults, turbines


def _dataset_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of a dataset for reproducibility."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(str(sorted(df.columns.tolist())).encode())
    # Sample first + last 100 rows' hash
    for col in df.columns[:10]:
        try:
            h.update(str(df[col].head(100).sum()).encode())
        except Exception:
            pass
    return h.hexdigest()[:16]


def _build_labels(df: pd.DataFrame) -> np.ndarray:
    """Binary fault label: 1 where fault_event_id is not null."""
    return (df["fault_event_id"].notna()).astype(int).to_numpy()


def run_detection_task(task_id: str, model_name: str, scorer_fn,
                       **scorer_kwargs) -> dict:
    """
    Run a detection task. scorer_fn must accept a DataFrame and return
    an array of severity scores (higher = more anomalous).
    """
    task = get_task(task_id)
    if task.task_type != TaskType.DETECTION:
        raise ValueError(f"Task {task_id} is not a DETECTION task")

    # For single-dataset tasks, load once
    eval_dataset = task.eval_datasets[0]
    df, faults, turbines = load_harmonized(eval_dataset)

    # Split: chronological 60/40 for CARE (event is in later portion)
    if "CARE" in eval_dataset:
        # Use first 60% for training, rest for eval
        split = int(len(df) * 0.6)
        test_df = df.iloc[split:].reset_index(drop=True)
        train_df = df.iloc[:split].reset_index(drop=True)
    else:
        # Kelmarsh/Penmanshiel: use existing train/test split
        t_max = df["event_ts"].max()
        cutoff = t_max - pd.Timedelta(days=7)
        train_df = df[df["event_ts"] < cutoff].reset_index(drop=True)
        test_df = df[df["event_ts"] >= cutoff].reset_index(drop=True)

    # Score
    y_score = scorer_fn(train_df, test_df, **scorer_kwargs)
    y_true = _build_labels(test_df)

    # Metrics
    result_metrics = M.detection_metrics_with_ci(y_true, y_score, n_boot=200)

    return {
        "task_id": task_id,
        "task_type": task.task_type.value,
        "model_name": model_name,
        "dataset": eval_dataset,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_faults": int(y_true.sum()),
        "dataset_hash": _dataset_hash(df),
        "metrics": result_metrics,
    }


def run_transfer_task(task_id: str, model_name: str, scorer_fn,
                       **scorer_kwargs) -> dict:
    """Cross-site transfer: train on X, test on Y."""
    task = get_task(task_id)
    if task.task_type != TaskType.CROSS_SITE_TRANSFER:
        raise ValueError(f"Task {task_id} is not a CROSS_SITE_TRANSFER task")

    train_dataset = task.train_datasets[0]
    eval_dataset = task.eval_datasets[0]

    train_df, _, _ = load_harmonized(train_dataset)
    eval_df, _, _ = load_harmonized(eval_dataset)

    y_score = scorer_fn(train_df, eval_df, **scorer_kwargs)
    y_true = _build_labels(eval_df)

    result_metrics = M.detection_metrics(y_true, y_score)

    # In-domain reference AUC (if available in prior result)
    in_domain_ref = 0.9  # to be overridden if caller provides
    gap = M.transfer_gap(in_domain_ref, result_metrics.get("auc_roc", 0))

    return {
        "task_id": task_id,
        "task_type": task.task_type.value,
        "model_name": model_name,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "metrics": result_metrics,
        "transfer_analysis": gap,
    }


def save_result(result: dict, out_dir: Path = RESULTS_DIR):
    """Append result to leaderboard-compatible JSON log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{result['task_id']}__{result['model_name']}.json"
    with out_file.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    return out_file


def collect_results(task_filter: str = None) -> pd.DataFrame:
    """Scan results directory and return all leaderboard rows."""
    if not RESULTS_DIR.exists():
        return pd.DataFrame()
    rows = []
    for p in RESULTS_DIR.glob("*.json"):
        try:
            data = json.load(p.open())
            if task_filter and data.get("task_id", "") != task_filter:
                continue
            # Flatten metrics
            flat = {
                "task_id": data.get("task_id"),
                "model_name": data.get("model_name"),
                "dataset": data.get("dataset") or data.get("eval_dataset"),
                "n_test": data.get("n_test"),
                "n_faults": data.get("n_faults"),
            }
            for k, v in data.get("metrics", {}).items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}_{kk}"] = vv
                else:
                    flat[k] = v
            rows.append(flat)
        except Exception as e:
            print(f"[WARN] {p}: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Print current leaderboard
    df = collect_results()
    if df.empty:
        print("No results yet. Run a task first.")
    else:
        print(df.to_string())
