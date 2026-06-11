"""WindBench — Wind Turbine Condition Monitoring Benchmark."""

from src.benchmark.schema import (
    CANONICAL_RAW_FEATURES, PARQUET_COLUMNS,
    FaultSubsystem, FaultLabelType, TurbineMeta, FaultEvent, DatasetManifest,
)
from src.benchmark.registry import REGISTRY, get_manifest, list_available, list_pending
from src.benchmark.tasks import TASKS, TaskType, get_task, list_tasks

__version__ = "1.0.0"
