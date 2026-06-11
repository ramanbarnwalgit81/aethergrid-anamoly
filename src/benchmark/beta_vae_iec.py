"""
β-VAE with IEC 61400-disentangled latent axes.

FIRST β-VAE whose latent dimensions are aligned to IEC 61400 variables.

REFERENCES
----------
- Higgins et al. 2017. "β-VAE: Learning Basic Visual Concepts with a
  Constrained Variational Framework." ICLR.
- Locatello et al. 2019. "Challenging Common Assumptions in the Unsupervised
  Learning of Disentangled Representations." ICML.  (Key insight: explicit
  inductive bias is required for disentanglement; our IEC-61400 physics
  regularizer IS that bias.)

SETUP
-----
A standard β-VAE maps input x (all SCADA signals) to a latent z ∈ R^d.
We add a PHYSICS REGULARIZER that forces specific latent dimensions to
encode IEC 61400-12-1-named variables:

  - z[0] : wind_speed_ms (forced via MSE between z[0] and normalized wind speed)
  - z[1] : ambient_temp_c
  - z[2] : active_power_kw
  - z[3] : rotor_speed_rpm
  - z[4:d] : free latent dimensions (capture unknown degradation modes)

At test time: reconstruction error + latent-prior Mahalanobis distance is
the anomaly score. Disentanglement enables per-axis attribution: which
IEC variable caused the anomaly.

Usage:
    python -m src.benchmark.beta_vae_iec
"""

from pathlib import Path
import json
import os, sys
import time

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score


RESULTS_DIR = Path("docs/results")
CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")

# IEC 61400 variables → CARE Farm A sensor mapping
IEC_AXES = [
    ("wind_speed_3_avg", "wind_speed_ms"),
    ("sensor_0_avg",     "ambient_temp_c"),
    ("power_30_avg",     "active_power_kw"),
    ("sensor_18_avg",    "rotor_speed_rpm"),
]
N_IEC = len(IEC_AXES)


class BetaVAE(nn.Module):
    """
    β-VAE with IEC 61400-aligned first N_IEC latent dimensions.
    """
    def __init__(self, n_features: int, latent_dim: int = 12, hidden: int = 64):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.enc1 = nn.Linear(n_features, hidden)
        self.enc2 = nn.Linear(hidden, hidden // 2)
        self.enc_mu = nn.Linear(hidden // 2, latent_dim)
        self.enc_logvar = nn.Linear(hidden // 2, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden // 2)
        self.dec2 = nn.Linear(hidden // 2, hidden)
        self.dec_out = nn.Linear(hidden, n_features)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return self.dec_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def load_event(event_id: int) -> pd.DataFrame:
    return pd.read_csv(CARE_DIR / "datasets" / f"{event_id}.csv",
                         sep=";", low_memory=False)


def build_pooled_training():
    """Collect train_test=='train' AND status_type_id==0 rows from normal events."""
    events = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    feature_cols = [s[0] for s in IEC_AXES] + [
        "sensor_11_avg", "sensor_13_avg", "sensor_14_avg",  # bearings
        "sensor_15_avg", "sensor_7_avg",                     # winding, nacelle
    ]

    X_parts = []
    iec_parts = []
    for _, row in events.iterrows():
        if row["event_label"] != "normal":
            continue
        try:
            df = load_event(int(row["event_id"]))
        except Exception:
            continue
        if "status_type_id" not in df.columns or "train_test" not in df.columns:
            continue
        mask = (df["train_test"] == "train") & (df["status_type_id"].fillna(1) == 0)
        if mask.sum() < 500:
            continue
        avail = [c for c in feature_cols if c in df.columns]
        if len(avail) < 4:
            continue
        X = df.loc[mask, avail].fillna(method="ffill").fillna(0).to_numpy(dtype=np.float32)
        X_parts.append(X)
        # IEC target values (what we want latent to match)
        iec_cols = [c for c, _ in IEC_AXES if c in df.columns]
        iec = df.loc[mask, iec_cols].fillna(method="ffill").fillna(0).to_numpy(dtype=np.float32)
        iec_parts.append(iec)

    if not X_parts:
        return None, None, None

    X_all = np.vstack(X_parts)
    iec_all = np.vstack(iec_parts)
    # Subsample to 20k rows for speed
    if len(X_all) > 20000:
        idx = np.random.RandomState(42).choice(len(X_all), 20000, replace=False)
        X_all = X_all[idx]; iec_all = iec_all[idx]
    return X_all, iec_all, feature_cols


def train_beta_vae(X_train: np.ndarray, iec_targets: np.ndarray,
                     beta: float = 4.0, lambda_iec: float = 5.0,
                     n_epochs: int = 50, batch_size: int = 128,
                     lr: float = 1e-3):
    torch.manual_seed(42)
    n_features = X_train.shape[1]
    model = BetaVAE(n_features=n_features, latent_dim=12, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Normalize IEC targets (to unit variance) so disentanglement loss is scale-sane
    iec_mean = iec_targets.mean(axis=0, keepdims=True)
    iec_std = iec_targets.std(axis=0, keepdims=True) + 1e-4
    iec_normalized = (iec_targets - iec_mean) / iec_std

    X = torch.FloatTensor(X_train)
    T = torch.FloatTensor(iec_normalized)
    n = len(X)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx = X[idx]; bt = T[idx]
            opt.zero_grad()
            recon, mu, logvar, z = model(bx)
            recon_loss = F.mse_loss(recon, bx, reduction="sum") / bx.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / bx.size(0)
            # IEC alignment: first N_IEC latent dims should match normalized IEC targets
            iec_align = F.mse_loss(mu[:, :N_IEC], bt)
            loss = recon_loss + beta * kl + lambda_iec * iec_align
            loss.backward()
            opt.step()
            total_loss += loss.item() * bx.size(0)
    model.eval()
    return model, iec_mean, iec_std


def score_event(model, df: pd.DataFrame, feature_cols: list,
                  iec_mean: np.ndarray, iec_std: np.ndarray) -> tuple:
    """Return (recon_score, latent_score, per_axis_contribution)."""
    avail = [c for c in feature_cols if c in df.columns]
    X = df[avail].fillna(method="ffill").fillna(0).to_numpy(dtype=np.float32)
    if X.shape[1] < model.n_features:
        pad = np.zeros((len(X), model.n_features - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > model.n_features:
        X = X[:, :model.n_features]

    with torch.no_grad():
        x_t = torch.FloatTensor(X)
        recon, mu, logvar, z = model(x_t)
        recon_err = ((x_t - recon) ** 2).mean(dim=1).numpy()
        # Latent anomaly: Mahalanobis-like score in latent space
        latent_dev = (mu ** 2).sum(dim=1).numpy()
        # Per-axis contribution on first N_IEC dims
        per_axis = {}
        for i, (_, name) in enumerate(IEC_AXES):
            per_axis[name] = (mu[:, i] ** 2).numpy()
    return recon_err, latent_dev, per_axis


def main():
    print("=" * 80)
    print("  β-VAE with IEC-61400 disentangled latents — CARE Farm A")
    print("  First IEC-named latent space in wind CMS")
    print("=" * 80)

    X_train, iec_targets, feature_cols = build_pooled_training()
    if X_train is None:
        print("[ERR] no training data")
        return

    print(f"\n[TRAIN] X_train shape {X_train.shape}, IEC targets {iec_targets.shape}")
    print(f"        features used: {feature_cols}")

    # Scale
    scaler = RobustScaler().fit(X_train)
    X_train_s = np.clip(scaler.transform(X_train), -10, 10).astype(np.float32)

    t0 = time.time()
    model, iec_mean, iec_std = train_beta_vae(
        X_train_s, iec_targets, beta=4.0, lambda_iec=5.0, n_epochs=40,
    )
    print(f"[TRAIN] done in {time.time()-t0:.0f}s")

    # Evaluate on anomaly events
    events = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    anomaly = events[events["event_label"] == "anomaly"]

    results = []
    for _, row in anomaly.iterrows():
        eid = int(row["event_id"])
        df = load_event(eid)
        n = len(df)

        # Need to scale input same way
        avail = [c for c in feature_cols if c in df.columns]
        if len(avail) < 4:
            continue
        X = df[avail].fillna(method="ffill").fillna(0).to_numpy(dtype=np.float32)
        # Pad to match training features
        if X.shape[1] < len(feature_cols):
            pad = np.zeros((len(X), len(feature_cols) - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        X_s = np.clip(scaler.transform(X[:, :len(feature_cols)]), -10, 10).astype(np.float32)

        with torch.no_grad():
            x_t = torch.FloatTensor(X_s)
            recon, mu, logvar, z = model(x_t)
            recon_err = ((x_t - recon) ** 2).mean(dim=1).numpy()
            latent_dev = (mu ** 2).sum(dim=1).numpy()

        # Labels
        y_event = np.zeros(n, dtype=int)
        s_idx = int(row.get("event_start_id", -1))
        e_idx = int(row.get("event_end_id", -1))
        if s_idx >= 0 and e_idx >= s_idx:
            y_event[s_idx:e_idx + 1] = 1

        # Test region
        test_mask = (df.get("train_test") == "prediction").to_numpy() if \
            "train_test" in df.columns else np.ones(n, dtype=bool)

        auc_recon = auc_latent = None
        if test_mask.sum() > 0:
            y_t = y_event[test_mask]
            if len(np.unique(y_t)) == 2:
                auc_recon = float(roc_auc_score(y_t, recon_err[test_mask]))
                auc_latent = float(roc_auc_score(y_t, latent_dev[test_mask]))

        # Per-axis IEC dev on test region
        per_axis_dev = {}
        for i, (_, name) in enumerate(IEC_AXES):
            per_axis_dev[name] = float(mu[test_mask, i].pow(2).mean().item()) if test_mask.sum() > 0 else 0.0

        print(f"  event {eid:>3} ({str(row['event_description'])[:25]:<25})  "
              f"recon AUC={auc_recon}  latent AUC={auc_latent}")

        results.append({
            "event_id": eid,
            "fault": str(row.get("event_description", ""))[:40],
            "auc_reconstruction": auc_recon,
            "auc_latent_prior": auc_latent,
            "per_iec_axis_deviation": per_axis_dev,
        })

    # Aggregate
    mean_recon = np.mean([r["auc_reconstruction"] for r in results
                            if r.get("auc_reconstruction") is not None])
    mean_latent = np.mean([r["auc_latent_prior"] for r in results
                             if r.get("auc_latent_prior") is not None])
    print(f"\n[AGGREGATE]")
    print(f"  Mean recon AUC: {mean_recon:.4f}")
    print(f"  Mean latent AUC: {mean_latent:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "beta_vae_iec.json"
    with out_path.open("w") as f:
        json.dump({
            "beta": 4.0,
            "lambda_iec": 5.0,
            "iec_axes": [name for _, name in IEC_AXES],
            "mean_auc_reconstruction": float(mean_recon),
            "mean_auc_latent_prior": float(mean_latent),
            "per_event": results,
        }, f, indent=2, default=str)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
