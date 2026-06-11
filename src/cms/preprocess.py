"""
Leak-free preprocessing: every statistic (kept columns, medians, scaler) is fit
on a dataset's TRAIN frame only and then applied to the prediction frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    feature_cols: List[str]
    kept_cols: Optional[List[str]] = None
    medians: Optional[np.ndarray] = None
    scaler: Optional[StandardScaler] = None

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        X = train_df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        # Drop columns that are all-NaN or (near-)constant in TRAIN.
        keep = []
        for c in self.feature_cols:
            col = X[c]
            if col.notna().sum() < 50:
                continue
            if col.std(skipna=True) < 1e-9:
                continue
            keep.append(c)
        self.kept_cols = keep
        Xk = X[keep]
        self.medians = Xk.median(axis=0).to_numpy()
        Xi = Xk.fillna(Xk.median(axis=0)).to_numpy(dtype=np.float64)
        self.scaler = StandardScaler().fit(Xi)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.kept_cols is not None and self.scaler is not None
        X = df[self.kept_cols].apply(pd.to_numeric, errors="coerce")
        Xi = X.to_numpy(dtype=np.float64)
        # Impute with TRAIN medians (column-aligned).
        inds = np.where(np.isnan(Xi))
        if inds[0].size:
            Xi[inds] = np.take(self.medians, inds[1])
        return self.scaler.transform(Xi)

    @property
    def n_features(self) -> int:
        return len(self.kept_cols) if self.kept_cols else 0
