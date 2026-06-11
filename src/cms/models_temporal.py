"""
Temporal Normal-Behavior Model (TemporalNBM).

Reconstruction detectors (PCA, autoencoders — including the published CARE 0.66
baseline) score each 10-min timestamp independently and therefore miss *slow
degradation*: a bearing that warms over days stays within the instantaneous
normal manifold long after its trajectory has become abnormal.

TemporalNBM injects dynamics cheaply and robustly:
  1. project standardized features to the dominant PCA modes (keeps Farm C's 950
     features tractable and denoises),
  2. augment each mode with two physically-motivated temporal signals —
     deviation from its recent rolling mean (drift) and recent rolling std
     (volatility/instability) over multiple horizons,
  3. model the joint normal distribution of this augmented state with a
     Ledoit-Wolf Mahalanobis detector.

Everything is fit on TRAIN rows only. Rolling features are causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Horizons in 10-min steps: 6 = 1 h, 36 = 6 h, 144 = 24 h.
WINDOWS = (6, 36, 144)


def _temporal_augment(Z: np.ndarray, windows=WINDOWS) -> np.ndarray:
    """Append drift (deviation from recent mean) and recent volatility per column."""
    df = pd.DataFrame(Z)
    parts = [df.to_numpy()]
    for w in windows:
        roll = df.rolling(w, min_periods=1)
        drift = (df - roll.mean()).to_numpy()
        vol = roll.std().fillna(0.0).to_numpy()
        parts.append(drift)
        parts.append(vol)
    return np.hstack(parts).astype(np.float64)


class TemporalNBM:
    def __init__(self, reduce_above: int = 120, reduce_to: int = 60,
                 windows=WINDOWS):
        self.reduce_above = reduce_above
        self.reduce_to = reduce_to
        self.windows = windows
        self.pca: PCA | None = None
        self.scaler: StandardScaler | None = None
        self.cov: LedoitWolf | None = None

    def _project(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X) if self.pca is not None else X

    def fit(self, X: np.ndarray) -> "TemporalNBM":
        if X.shape[1] > self.reduce_above:
            k = min(self.reduce_to, X.shape[1], max(2, X.shape[0] - 1))
            self.pca = PCA(n_components=k, random_state=42).fit(X)
        Z = self._project(X)
        A = _temporal_augment(Z, self.windows)
        self.scaler = StandardScaler().fit(A)
        self.cov = LedoitWolf().fit(self.scaler.transform(A))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self._project(X)
        A = self.scaler.transform(_temporal_augment(Z, self.windows))
        return self.cov.mahalanobis(A)


TEMPORAL_SCORERS = {"TemporalNBM": TemporalNBM}
