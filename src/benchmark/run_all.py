"""
WindBench Grand Evaluation — runs every baseline on every available task
and saves leaderboard-ready JSON results.

Usage:  python -m src.benchmark.run_all
"""

import json
import time
from pathlib import Path

import pandas as pd

from src.benchmark.baselines import BASELINES
from src.benchmark.registry import REGISTRY, list_available
from src.benchmark.tasks import TASKS, TaskType
from src.benchmark.evaluate import (
    run_detection_task, run_transfer_task, save_result, collect_results,
)

RESULTS_DIR = Path("data/benchmark/results")


def run_detection_grid():
    """Run every baseline on every available detection task."""
    available = list_available()
    print(f"[WindBench] Available harmonized datasets: {available}")

    detection_tasks = [tid for tid, t in TASKS.items()
                        if t.task_type == TaskType.DETECTION
                        and all(d in available for d in t.eval_datasets)]

    print(f"[WindBench] Detection tasks to run: {detection_tasks}")
    print(f"[WindBench] Baselines to run: {list(BASELINES.keys())}")

    total = len(detection_tasks) * len(BASELINES)
    done = 0

    for task_id in detection_tasks:
        for model_name, scorer_fn in BASELINES.items():
            done += 1
            print(f"\n[{done}/{total}] Task: {task_id}  Model: {model_name}")
            t0 = time.time()
            try:
                result = run_detection_task(task_id, model_name, scorer_fn)
                result["latency_seconds"] = round(time.time() - t0, 2)
                save_result(result, RESULTS_DIR)
                m = result["metrics"]
                auc_ci = m.get("auc_roc_ci", {})
                print(f"    AUC-ROC: {m.get('auc_roc', 0):.4f} "
                      f"[95% CI: {auc_ci.get('ci_lower', 0):.4f}, {auc_ci.get('ci_upper', 0):.4f}]  "
                      f"n_faults={result['n_faults']}  "
                      f"time={result['latency_seconds']}s")
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")


def run_transfer_grid():
    """Cross-site transfer tasks."""
    available = list_available()
    transfer_tasks = [tid for tid, t in TASKS.items()
                       if t.task_type == TaskType.CROSS_SITE_TRANSFER
                       and t.train_datasets[0] in available
                       and t.eval_datasets[0] in available]

    print(f"\n[WindBench] Transfer tasks: {transfer_tasks}")

    for task_id in transfer_tasks:
        for model_name, scorer_fn in BASELINES.items():
            print(f"\n[Transfer] {task_id} — {model_name}")
            t0 = time.time()
            try:
                result = run_transfer_task(task_id, model_name, scorer_fn)
                result["latency_seconds"] = round(time.time() - t0, 2)
                save_result(result, RESULTS_DIR)
                m = result["metrics"]
                print(f"    AUC-ROC (transfer): {m.get('auc_roc', 0):.4f}")
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")


def print_leaderboard():
    df = collect_results()
    if df.empty:
        print("No results.")
        return
    print("\n" + "=" * 90)
    print("  WINDBENCH LEADERBOARD (sorted by AUC-ROC per task)")
    print("=" * 90)
    for task_id in df["task_id"].dropna().unique():
        sub = df[df["task_id"] == task_id].sort_values("auc_roc", ascending=False)
        print(f"\n--- {task_id} ---")
        cols = ["model_name", "auc_roc", "auc_pr", "recall_at_fpr_0.10", "f1_best",
                "n_faults"]
        cols = [c for c in cols if c in sub.columns]
        print(sub[cols].to_string(index=False))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_detection_grid()
    run_transfer_grid()
    print_leaderboard()

    # Save flat CSV leaderboard
    df = collect_results()
    if not df.empty:
        df.to_csv("data/benchmark/leaderboard.csv", index=False)
        print(f"\n[OK] Saved flat leaderboard: data/benchmark/leaderboard.csv")


if __name__ == "__main__":
    main()
