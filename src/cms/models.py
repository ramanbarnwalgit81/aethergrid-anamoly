"""
Unsupervised normal-behavior scorers.

Every scorer implements the same contract:
    fit(X_train)            # X_train: standardized normal-operation matrix
    score(X) -> np.ndarray  # higher = more anomalous

These work on any farm (A/B/C) with no semantic feature mapping — the published
Autoencoder baseline (CARE 0.66) is exactly a reconstruction scorer of this kind.
Deep temporal models live in models_deep.py to keep the torch dependency optional.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class PCAReconstruction:
    """Reconstruction error from a low-rank PCA of normal behavior."""

    def __init__(self, var: float = 0.95, max_components: int = 40):
        self.var = var
        self.max_components = max_components
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCAReconstruction":
        k = min(self.max_components, X.shape[1], max(1, X.shape[0] - 1))
        self.pca = PCA(n_components=k, random_state=42).fit(X)
        # Trim to components explaining `var` of variance.
        cum = np.cumsum(self.pca.explained_variance_ratio_)
        keep = int(np.searchsorted(cum, self.var) + 1)
        self.pca = PCA(n_components=max(1, min(keep, k)), random_state=42).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.pca.transform(X)
        Xr = self.pca.inverse_transform(Z)
        return np.sqrt(np.mean((X - Xr) ** 2, axis=1))


class MahalanobisScorer:
    """Mahalanobis distance to the normal-operation distribution (Ledoit-Wolf).

    For high-dimensional farms (Farm C has ~950 features) a full p×p shrinkage
    covariance + inversion is expensive and ill-conditioned, so we first project
    to a PCA subspace (retaining `reduce_to` components) when p exceeds
    `reduce_above`. This is both faster and more robust, and is standard practice.
    """

    def __init__(self, reduce_above: int = 150, reduce_to: int = 80):
        self.reduce_above = reduce_above
        self.reduce_to = reduce_to
        self.pca: PCA | None = None
        self.cov: LedoitWolf | None = None

    def fit(self, X: np.ndarray) -> "MahalanobisScorer":
        if X.shape[1] > self.reduce_above:
            k = min(self.reduce_to, X.shape[1], max(2, X.shape[0] - 1))
            self.pca = PCA(n_components=k, random_state=42).fit(X)
            X = self.pca.transform(X)
        self.cov = LedoitWolf().fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.pca is not None:
            X = self.pca.transform(X)
        return self.cov.mahalanobis(X)


class IForestScorer:
    def __init__(self, n_estimators: int = 300, contamination: str | float = "auto"):
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            max_samples="auto", random_state=42, n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> "IForestScorer":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # score_samples: higher = more normal -> negate for "higher = anomalous".
        return -self.model.score_samples(X)


class KNNScorer:
    """Mean distance to k nearest training neighbours (classic distance AD)."""

    def __init__(self, k: int = 20, max_train: int = 8000):
        self.k = k
        self.max_train = max_train
        self.nn = None

    def fit(self, X: np.ndarray) -> "KNNScorer":
        from sklearn.neighbors import NearestNeighbors
        if len(X) > self.max_train:
            X = X[np.random.default_rng(42).choice(len(X), self.max_train, replace=False)]
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d, _ = self.nn.kneighbors(X)
        return d.mean(axis=1)


class LOFScorer:
    """Local Outlier Factor in novelty mode."""

    def __init__(self, k: int = 20, max_train: int = 8000):
        self.k = k
        self.max_train = max_train
        self.model = None

    def fit(self, X: np.ndarray) -> "LOFScorer":
        from sklearn.neighbors import LocalOutlierFactor
        if len(X) > self.max_train:
            X = X[np.random.default_rng(42).choice(len(X), self.max_train, replace=False)]
        self.model = LocalOutlierFactor(n_neighbors=self.k, novelty=True).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


class ECODScorer:
    """Empirical-CDF Outlier Detection (Li et al. 2022): parameter-free, strong.

    Per feature, aggregate negative log empirical tail probabilities; the final
    score is the max of left-tail, right-tail, and skewness-directed aggregates.
    """

    def __init__(self):
        self.Xtr = None
        self.skew = None

    def fit(self, X: np.ndarray) -> "ECODScorer":
        self.Xtr = np.sort(np.asarray(X, dtype=np.float64), axis=0)
        from scipy.stats import skew
        self.skew = skew(np.asarray(X, dtype=np.float64), axis=0)
        return self

    def _tail(self, X, right=False):
        n = self.Xtr.shape[0]
        out = np.empty_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            col = self.Xtr[:, j]
            if right:
                # P(>= x) = (n - searchsorted_left)/n
                cnt = n - np.searchsorted(col, X[:, j], side="left")
            else:
                cnt = np.searchsorted(col, X[:, j], side="right")
            out[:, j] = -np.log(np.clip(cnt / n, 1.0 / n, 1.0))
        return out

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        ol = self._tail(X, right=False).sum(axis=1)
        orr = self._tail(X, right=True).sum(axis=1)
        # skew-directed: left tail for right-skewed dims, right tail otherwise
        tl = self._tail(X, right=False)
        tr = self._tail(X, right=True)
        oauto = np.where(self.skew[None, :] < 0, tl, tr).sum(axis=1)
        return np.maximum.reduce([ol, orr, oauto])


class OCSVMScorer:
    def __init__(self, nu: float = 0.05, gamma: str | float = "scale",
                 max_train: int = 8000):
        self.model = OneClassSVM(nu=nu, gamma=gamma, kernel="rbf")
        self.max_train = max_train

    def fit(self, X: np.ndarray) -> "OCSVMScorer":
        if len(X) > self.max_train:
            idx = np.random.default_rng(42).choice(len(X), self.max_train, replace=False)
            X = X[idx]
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


class EnsembleScorer:
    """Train-normalized average of several scorers (leak-free fusion).

    Each sub-scorer is z-normalized by its own TRAIN score distribution before
    averaging, so heterogeneous score scales combine fairly. Default members
    span a linear subspace model (PCA), a covariance model (Mahalanobis), and a
    nonlinear reconstruction model (DenseAE).
    """

    def __init__(self, members=("PCARecon", "Mahalanobis")):
        self.members = list(members)
        self.subs = []
        self.mu = []
        self.sd = []

    def fit(self, X):
        self.subs, self.mu, self.sd = [], [], []
        for name in self.members:
            s = make_scorer(name)
            s.fit(X)
            tr = s.score(X)
            self.subs.append(s)
            self.mu.append(float(np.mean(tr)))
            self.sd.append(float(np.std(tr)) + 1e-9)
        return self

    def score(self, X):
        zs = [(s.score(X) - mu) / sd for s, mu, sd in zip(self.subs, self.mu, self.sd)]
        return np.mean(np.vstack(zs), axis=0)


SCORERS = {
    "PCARecon": PCAReconstruction,
    "Mahalanobis": MahalanobisScorer,
    "IsolationForest": IForestScorer,
    "OCSVM": OCSVMScorer,
    "KNN": KNNScorer,
    "LOF": LOFScorer,
    "ECOD": ECODScorer,
    "Ensemble": EnsembleScorer,
}


def make_scorer(name: str):
    if name in SCORERS:
        return SCORERS[name]()
    from src.cms.models_temporal import TEMPORAL_SCORERS
    if name in TEMPORAL_SCORERS:
        return TEMPORAL_SCORERS[name]()
    from src.cms.models_graph import GRAPH_SCORERS
    if name in GRAPH_SCORERS:
        return GRAPH_SCORERS[name]()
    # Deep scorers live in models_deep so torch stays an optional dependency.
    from src.cms.models_deep import DEEP_SCORERS
    if name in DEEP_SCORERS:
        return DEEP_SCORERS[name]()
    raise ValueError(f"Unknown scorer {name}; available: "
                     f"{list(SCORERS)} + temporal {list(TEMPORAL_SCORERS)} "
                     f"+ deep {list(DEEP_SCORERS)}")
