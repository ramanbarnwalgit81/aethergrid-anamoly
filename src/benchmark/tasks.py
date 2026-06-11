"""
WindBench Standardized Tasks.

Defines the five tasks every model is evaluated on. Standardized protocol
ensures cross-paper comparability.

Tasks:
  1. DETECTION            — classify fault vs normal periods
  2. RUL                  — predict remaining useful life in hours
  3. CROSS_SITE_TRANSFER  — train on farm X, test on farm Y (zero-shot)
  4. ADVERSARIAL          — robustness under FGSM/PGD/physical attacks
  5. EARLY_WARNING        — time-to-detection (how early is the alert?)

Each task has a standard split protocol and evaluation metric.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional


class TaskType(Enum):
    DETECTION = "detection"
    RUL = "rul"
    CROSS_SITE_TRANSFER = "cross_site_transfer"
    ADVERSARIAL = "adversarial"
    EARLY_WARNING = "early_warning"


@dataclass
class TaskDefinition:
    task_id: str
    task_type: TaskType
    train_datasets: List[str]        # dataset_ids used for training
    eval_datasets: List[str]         # dataset_ids used for evaluation
    primary_metric: str              # e.g., "auc_roc", "rul_mae_hours"
    secondary_metrics: List[str]
    description: str
    scoring_protocol: str = "standard"


# ═══════════════════════════════════════════════════════════════════
# CANONICAL TASKS
# ═══════════════════════════════════════════════════════════════════

TASKS = {
    # ── Task 1: Detection on real labels ──
    "T1_detection_care_a": TaskDefinition(
        task_id="T1_detection_care_a",
        task_type=TaskType.DETECTION,
        train_datasets=["CARE_FARM_A"],
        eval_datasets=["CARE_FARM_A"],
        primary_metric="auc_roc",
        secondary_metrics=["auc_pr", "recall_at_fpr_0.10", "f1_best"],
        description=("Anomaly detection on CARE Farm A real labeled faults. "
                     "Train on pre-event data, evaluate on event period. "
                     "Aggregated across 11 anomaly events."),
    ),

    "T1_detection_kelmarsh": TaskDefinition(
        task_id="T1_detection_kelmarsh",
        task_type=TaskType.DETECTION,
        train_datasets=["KELMARSH"],
        eval_datasets=["KELMARSH"],
        primary_metric="auc_roc",
        secondary_metrics=["auc_pr", "recall_at_fpr_0.10", "f1_best"],
        description=("Anomaly detection on Kelmarsh with injected faults. "
                     "80/20 chronological split."),
    ),

    "T1_detection_all": TaskDefinition(
        task_id="T1_detection_all",
        task_type=TaskType.DETECTION,
        train_datasets=["KELMARSH", "PENMANSHIEL", "CARE_FARM_A"],
        eval_datasets=["KELMARSH", "PENMANSHIEL", "CARE_FARM_A"],
        primary_metric="auc_roc",
        secondary_metrics=["auc_pr", "recall_at_fpr_0.10"],
        description=("Multi-farm detection — mean AUC across all farms."),
    ),

    # ── Task 2: Remaining Useful Life ──
    "T2_rul_care_a": TaskDefinition(
        task_id="T2_rul_care_a",
        task_type=TaskType.RUL,
        train_datasets=["CARE_FARM_A"],
        eval_datasets=["CARE_FARM_A"],
        primary_metric="rul_mae_hours",
        secondary_metrics=["rul_rmse_hours", "rul_within_24h_fraction", "rul_rho_spearman"],
        description=("RUL prediction on CARE Farm A. Train on pre-event rows, "
                     "evaluate on event period rows."),
    ),

    # ── Task 3: Cross-Site Transfer ──
    "T3_transfer_kelmarsh_to_penmanshiel": TaskDefinition(
        task_id="T3_transfer_kelmarsh_to_penmanshiel",
        task_type=TaskType.CROSS_SITE_TRANSFER,
        train_datasets=["KELMARSH"],
        eval_datasets=["PENMANSHIEL"],
        primary_metric="auc_roc",
        secondary_metrics=["auc_pr", "transfer_gap"],  # gap = in-domain - out-of-domain
        description=("Zero-shot transfer: train on Kelmarsh, test on Penmanshiel."),
    ),

    "T3_transfer_care_a_to_care_c": TaskDefinition(
        task_id="T3_transfer_care_a_to_care_c",
        task_type=TaskType.CROSS_SITE_TRANSFER,
        train_datasets=["CARE_FARM_A"],
        eval_datasets=["CARE_FARM_C"],
        primary_metric="auc_roc",
        secondary_metrics=["auc_pr", "transfer_gap"],
        description=("Onshore Portugal → Offshore Germany zero-shot."),
    ),

    # ── Task 4: Adversarial Robustness ──
    "T4_adversarial_kelmarsh": TaskDefinition(
        task_id="T4_adversarial_kelmarsh",
        task_type=TaskType.ADVERSARIAL,
        train_datasets=["KELMARSH"],
        eval_datasets=["KELMARSH"],
        primary_metric="adversarial_detection_rate",
        secondary_metrics=["fgsm_robustness", "pgd_robustness", "physical_robustness"],
        description=("Detection rate under FGSM/PGD/physical attacks. "
                     "Higher = more robust."),
    ),

    # ── Task 5: Early Warning ──
    "T5_early_warning_care_a": TaskDefinition(
        task_id="T5_early_warning_care_a",
        task_type=TaskType.EARLY_WARNING,
        train_datasets=["CARE_FARM_A"],
        eval_datasets=["CARE_FARM_A"],
        primary_metric="early_warning_fraction",
        secondary_metrics=["time_to_detection_hours", "false_alarm_rate"],
        description=("How many hours BEFORE event_start_id does the model fire? "
                     "Higher = more useful for maintenance planning."),
    ),
}


def list_tasks() -> List[str]:
    return list(TASKS.keys())


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list_tasks()}")
    return TASKS[task_id]
