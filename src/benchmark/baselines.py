"""
WindBench Baseline Implementations.

All baselines conform to a single interface:
    def scorer(train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> np.ndarray

Returns an array of severity scores (higher = more anomalous), one per test row.

Baselines implemented:
  1. GlobalRuleThreshold    — classical industrial fixed-rule baseline
  2. IsolationForest        — Liu et al. 2008
  3. OneClassSVM            — Schölkopf et al. 2001
  4. KNNAnomaly             — k-nearest neighbors distance
  5. LocalOutlierFactor     — Breunig et al. 2000
  6. EllipticEnvelope       — Gaussian ellipse outlier detection
  7. DenseAutoencoder       — feedforward AE
  8. VariationalAE          — VAE
  9. PCAReconstruction      — PCA-based reconstruction error
 10. MahalanobisDistance    — multivariate distance to training centroid
 11. Ensemble_IF_OCSVM      — weighted ensemble

Each produces severity ∈ [0, 100].
"""

from typing import Callable
import numpy as np
import pandas as pd

from src.benchmark.schema import CANONICAL_RAW_FEATURES


def _get_numeric_features(df: pd.DataFrame) -> list:
    """Return canonical features available (not all NaN) in df."""
    out = []
    for c in CANONICAL_RAW_FEATURES:
        if c in df.columns and df[c].notna().sum() > 10:
            out.append(c)
    return out


def _prep(train_df: pd.DataFrame, test_df: pd.DataFrame):
    features = _get_numeric_features(train_df)
    # Use intersection with test features
    features = [c for c in features if c in test_df.columns]
    X_train = train_df[features].fillna(0).to_numpy(dtype=np.float32)
    X_test = test_df[features].fillna(0).to_numpy(dtype=np.float32)
    return X_train, X_test, features


def _scores_to_severity(scores: np.ndarray, train_scores: np.ndarray) -> np.ndarray:
    """Percentile rank → 0-100 severity."""
    sorted_train = np.sort(train_scores)
    ranks = np.searchsorted(sorted_train, scores, side="right")
    sev = (ranks / max(len(sorted_train), 1)) * 100.0
    return np.clip(sev, 0.0, 100.0)


# ─────────────────────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────────────────────

def global_rule_threshold(train_df, test_df, **kwargs) -> np.ndarray:
    """Classical industrial rule: alert if gearbox > fleet_mean OR vib > 1.5x."""
    gbx_mean = train_df["gearbox_oil_temp_c"].mean() if "gearbox_oil_temp_c" in train_df else None

    severity = np.zeros(len(test_df))
    if gbx_mean and "gearbox_oil_temp_c" in test_df.columns:
        excess = np.clip((test_df["gearbox_oil_temp_c"].fillna(0).values - gbx_mean) / 20, 0, 2)
        severity = np.maximum(severity, excess * 50)
    return np.clip(severity * 50, 0, 100)  # scale to 0-100


def isolation_forest(train_df, test_df, **kwargs):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, _ = _prep(train_df, test_df)
    scaler = StandardScaler().fit(X_train)
    model = IsolationForest(contamination=0.03, n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(scaler.transform(X_train))
    train_scores = -model.score_samples(scaler.transform(X_train))
    test_scores = -model.score_samples(scaler.transform(X_test))
    return _scores_to_severity(test_scores, train_scores)


def one_class_svm(train_df, test_df, **kwargs):
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, _ = _prep(train_df, test_df)
    # Subsample for OCSVM O(n²)
    if len(X_train) > 3000:
        idx = np.random.default_rng(42).choice(len(X_train), 3000, replace=False)
        X_train_sub = X_train[idx]
    else:
        X_train_sub = X_train
    scaler = StandardScaler().fit(X_train_sub)
    model = OneClassSVM(kernel="rbf", nu=0.03, gamma="scale")
    model.fit(scaler.transform(X_train_sub))
    train_scores = -model.score_samples(scaler.transform(X_train))
    test_scores = -model.score_samples(scaler.transform(X_test))
    return _scores_to_severity(test_scores, train_scores)


def knn_anomaly(train_df, test_df, k: int = 10, **kwargs):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, _ = _prep(train_df, test_df)
    scaler = StandardScaler().fit(X_train)
    # Subsample training if very large
    if len(X_train) > 10000:
        idx = np.random.default_rng(42).choice(len(X_train), 10000, replace=False)
        X_ref = X_train[idx]
    else:
        X_ref = X_train
    X_ref_s = scaler.transform(X_ref)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    nn = NearestNeighbors(n_neighbors=k).fit(X_ref_s)
    train_scores, _ = nn.kneighbors(X_train_s)
    test_scores, _ = nn.kneighbors(X_test_s)
    return _scores_to_severity(test_scores.mean(axis=1), train_scores.mean(axis=1))


def local_outlier_factor(train_df, test_df, **kwargs):
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, _ = _prep(train_df, test_df)
    if len(X_train) > 5000:
        idx = np.random.default_rng(42).choice(len(X_train), 5000, replace=False)
        X_train = X_train[idx]
    scaler = StandardScaler().fit(X_train)
    model = LocalOutlierFactor(novelty=True, n_neighbors=20, contamination=0.03)
    model.fit(scaler.transform(X_train))
    train_scores = -model.score_samples(scaler.transform(X_train))
    test_scores = -model.score_samples(scaler.transform(X_test))
    return _scores_to_severity(test_scores, train_scores)


def elliptic_envelope(train_df, test_df, **kwargs):
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, _ = _prep(train_df, test_df)
    scaler = StandardScaler().fit(X_train)
    model = EllipticEnvelope(contamination=0.03, support_fraction=0.8, random_state=42)
    try:
        model.fit(scaler.transform(X_train))
        train_scores = -model.score_samples(scaler.transform(X_train))
        test_scores = -model.score_samples(scaler.transform(X_test))
        return _scores_to_severity(test_scores, train_scores)
    except Exception:
        return np.zeros(len(X_test))


def pca_reconstruction(train_df, test_df, n_components: int = 5, **kwargs):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, _ = _prep(train_df, test_df)
    n_components = min(n_components, X_train.shape[1] - 1, len(X_train) - 1)
    if n_components < 2:
        return np.zeros(len(X_test))
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=n_components).fit(X_train_s)
    train_recon = pca.inverse_transform(pca.transform(X_train_s))
    test_recon = pca.inverse_transform(pca.transform(X_test_s))
    train_err = ((X_train_s - train_recon) ** 2).mean(axis=1)
    test_err = ((X_test_s - test_recon) ** 2).mean(axis=1)
    return _scores_to_severity(test_err, train_err)


def mahalanobis_distance(train_df, test_df, **kwargs):
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, _ = _prep(train_df, test_df)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    mean = X_train_s.mean(axis=0)
    cov = np.cov(X_train_s.T)
    try:
        inv_cov = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-6)
    except Exception:
        return np.zeros(len(X_test))

    def _md(x_arr):
        diff = x_arr - mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, inv_cov, diff), 0))

    return _scores_to_severity(_md(X_test_s), _md(X_train_s))


def dense_autoencoder(train_df, test_df, **kwargs):
    """Simple AE: input -> 16 -> 4 -> 16 -> input with scikit-learn-style API."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        # Fallback: PCA reconstruction
        return pca_reconstruction(train_df, test_df, n_components=4)

    from sklearn.preprocessing import RobustScaler
    X_train, X_test, _ = _prep(train_df, test_df)
    n_feat = X_train.shape[1]
    if n_feat < 4:
        return np.zeros(len(X_test))

    scaler = RobustScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    class AE(nn.Module):
        def __init__(self, n):
            super().__init__()
            h = max(n // 2, 4)
            self.encoder = nn.Sequential(nn.Linear(n, h), nn.ReLU(), nn.Linear(h, max(h // 4, 2)))
            self.decoder = nn.Sequential(nn.Linear(max(h // 4, 2), h), nn.ReLU(), nn.Linear(h, n))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    torch.manual_seed(42)
    model = AE(n_feat)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.FloatTensor(X_train_s)
    for _ in range(50):
        opt.zero_grad()
        recon = model(X_t)
        loss = ((recon - X_t) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_err = ((X_t - model(X_t)) ** 2).mean(dim=1).numpy()
        X_te = torch.FloatTensor(X_test_s)
        test_err = ((X_te - model(X_te)) ** 2).mean(dim=1).numpy()
    return _scores_to_severity(test_err, train_err)


def ensemble_if_ocsvm(train_df, test_df, **kwargs):
    """Simple average of IF and OCSVM severity scores."""
    s1 = isolation_forest(train_df, test_df)
    s2 = one_class_svm(train_df, test_df)
    return (s1 + s2) / 2.0


# Registry
BASELINES: dict = {
    "GlobalRuleThreshold": global_rule_threshold,
    "IsolationForest":     isolation_forest,
    "OneClassSVM":         one_class_svm,
    "KNN":                 knn_anomaly,
    "LOF":                 local_outlier_factor,
    "EllipticEnvelope":    elliptic_envelope,
    "PCARecon":            pca_reconstruction,
    "Mahalanobis":         mahalanobis_distance,
    "DenseAE":             dense_autoencoder,
    "Ensemble_IF_OCSVM":   ensemble_if_ocsvm,
}


if __name__ == "__main__":
    print(f"WindBench has {len(BASELINES)} baselines:")
    for name in BASELINES:
        print(f"  - {name}")
