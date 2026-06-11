"""
CARE dataset loader — correct, feature-preserving, leak-free.

Each CARE dataset is ONE turbine's time series containing ONE event (an
``anomaly`` leading to a failure, or a ``normal`` reference period). The data
ships a ``train_test`` column splitting each file into a guaranteed-normal
``train`` history (~52k rows) and a ``prediction`` window (~2-3k rows) that
contains the event. The official protocol is: fit a Normal-Behavior Model on the
``train`` rows of a dataset, score its ``prediction`` rows, and evaluate at the
*event* level across all datasets in a farm.

Unlike the legacy adapter, this loader:
  * keeps ALL numeric sensor features (84 / 255 / 955 for farms A / B / C),
  * preserves the shipped ``train_test`` split and ``status_type_id``,
  * locates the labelled event window via the ``id`` column (not positional
    iloc), and
  * never pools independent datasets together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

CARE_BASE = Path("data/real_scada/care/extracted")

# Columns present in every CARE file that are NOT sensor features.
META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]

# status_type_id semantics (from the CARE README):
#   0 Normal production, 1 Derated, 2 Idling, 3 Service, 4 Downtime, 5 Other.
# 0 and 2 are "normal operation"; we additionally keep 1 (derated) as it is
# still power production. 3/4/5 are filtered when ``producing_only=True``.
PRODUCING_STATUS = {0, 1, 2}

FARMS = ["A", "B", "C"]


def _farm_dir(farm: str) -> Path:
    return CARE_BASE / f"Wind Farm {farm}" / f"Wind Farm {farm}"


@dataclass
class CareDataset:
    """One turbine-event dataset under the official CARE protocol."""

    farm: str
    event_id: int
    label: str  # "anomaly" | "normal"
    description: str
    event_start_id: int
    event_end_id: int
    path: Path
    df: pd.DataFrame = field(repr=False)
    feature_cols: List[str] = field(repr=False)

    # ----- masks -----------------------------------------------------------
    @cached_property
    def train_mask(self) -> np.ndarray:
        return (self.df["train_test"].astype(str).str.lower() == "train").to_numpy()

    @cached_property
    def predict_mask(self) -> np.ndarray:
        return (self.df["train_test"].astype(str).str.lower() == "prediction").to_numpy()

    @cached_property
    def event_mask(self) -> np.ndarray:
        """Per-row label: True inside the operator-labelled fault window.

        Only meaningful for anomaly datasets; all-False for normal datasets.
        Located via the ``id`` column to be robust to any row reordering.
        """
        ids = self.df["id"].to_numpy()
        if self.label != "anomaly":
            return np.zeros(len(ids), dtype=bool)
        return (ids >= self.event_start_id) & (ids <= self.event_end_id)

    # ----- feature matrices ------------------------------------------------
    def _matrix(self, mask: np.ndarray, producing_only: bool) -> pd.DataFrame:
        sub = self.df.loc[mask]
        if producing_only and "status_type_id" in sub.columns:
            sub = sub[sub["status_type_id"].isin(PRODUCING_STATUS)]
        return sub

    def train_frame(self, producing_only: bool = True) -> pd.DataFrame:
        return self._matrix(self.train_mask, producing_only)

    def predict_frame(self, producing_only: bool = False) -> pd.DataFrame:
        # Default keep all prediction rows: a fault can manifest as a stop.
        return self._matrix(self.predict_mask, producing_only)

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_predict(self) -> int:
        return int(self.predict_mask.sum())


def _numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    num = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    return num


def load_event(farm: str, event_id: int, event_row: pd.Series) -> Optional[CareDataset]:
    path = _farm_dir(farm) / "datasets" / f"{event_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";", low_memory=False)
    if "id" not in df.columns or "train_test" not in df.columns:
        return None
    # Parse timestamp (kept for ordering / earliness, not as a feature).
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.sort_values("id").reset_index(drop=True)
    feats = _numeric_feature_cols(df)
    return CareDataset(
        farm=farm,
        event_id=int(event_id),
        label=str(event_row["event_label"]).strip().lower(),
        description=str(event_row.get("event_description", "") or ""),
        event_start_id=int(event_row["event_start_id"]),
        event_end_id=int(event_row["event_end_id"]),
        path=path,
        df=df,
        feature_cols=feats,
    )


def iter_care(farm: str) -> Iterator[CareDataset]:
    """Yield every dataset of a farm, one at a time (memory-friendly)."""
    info_path = _farm_dir(farm) / "event_info.csv"
    if not info_path.exists():
        raise FileNotFoundError(f"event_info.csv missing for farm {farm}: {info_path}")
    info = pd.read_csv(info_path, sep=";")
    for _, row in info.iterrows():
        ds = load_event(farm, int(row["event_id"]), row)
        if ds is not None:
            yield ds


def common_feature_cols(farm: str) -> List[str]:
    """Feature columns present (numeric) in *every* dataset of a farm.

    CARE files within a farm share a schema, but defensively intersect so a
    fleet model can rely on a stable column set.
    """
    cols: Optional[set] = None
    for ds in iter_care(farm):
        s = set(ds.feature_cols)
        cols = s if cols is None else (cols & s)
    return sorted(cols) if cols else []


if __name__ == "__main__":
    # Reconnaissance / smoke test — prints real counts, fabricates nothing.
    for farm in FARMS:
        n_ds = n_anom = n_norm = 0
        feat_counts = []
        train_rows = predict_rows = event_rows = 0
        for ds in iter_care(farm):
            n_ds += 1
            n_anom += ds.label == "anomaly"
            n_norm += ds.label == "normal"
            feat_counts.append(len(ds.feature_cols))
            train_rows += ds.n_train
            predict_rows += ds.n_predict
            event_rows += int(ds.event_mask.sum())
        print(
            f"Farm {farm}: {n_ds} datasets ({n_anom} anomaly / {n_norm} normal), "
            f"features={min(feat_counts)}..{max(feat_counts)}, "
            f"train_rows={train_rows:,}, predict_rows={predict_rows:,}, "
            f"labelled_event_rows={event_rows:,}"
        )
