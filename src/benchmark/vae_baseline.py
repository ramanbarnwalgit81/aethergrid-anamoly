"""
Variational Autoencoder Baseline for WindBench.

Matches SOTA wind CMS methodology (Hybrid Autoencoder 2025, arxiv 2510.15010):
  - Encoder: input -> latent distribution (mu, log_sigma)
  - Reparameterization trick: z = mu + sigma * epsilon
  - Decoder: z -> reconstruction
  - Loss: reconstruction_MSE + beta * KL_divergence

Anomaly score combines reconstruction error and KL divergence:
  s(x) = reconstruction_MSE(x) + alpha * KL(q(z|x) || N(0, I))

This captures both:
  - "Does the model fail to reconstruct this input?" (reconstruction)
  - "Does this input lie outside the learned latent manifold?" (KL)

Both signals indicate anomaly.
"""

from pathlib import Path
import os, sys

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

# torch must import before numpy/pandas on Windows (OpenMP conflict).
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler


class VAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 8):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(n_features, hidden)
        self.enc2 = nn.Linear(hidden, hidden // 2)
        self.enc_mu = nn.Linear(hidden // 2, latent)
        self.enc_logvar = nn.Linear(hidden // 2, latent)
        # Decoder
        self.dec1 = nn.Linear(latent, hidden // 2)
        self.dec2 = nn.Linear(hidden // 2, hidden)
        self.dec_out = nn.Linear(hidden, n_features)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return self.dec_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 0.1):
    """MSE reconstruction + beta * KL divergence."""
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    # KL(q(z|x) || N(0, I))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl


def vae_anomaly_score(model, x, alpha: float = 0.3):
    """Combine reconstruction error and KL divergence as anomaly score."""
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(x)
        recon_err = ((x - recon) ** 2).mean(dim=-1)
        # Per-sample KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        # Normalize kl to reconstruction scale
        combined = recon_err + alpha * (kl / max(x.size(-1), 1))
        return combined.numpy()


def train_vae(X_train: np.ndarray, n_epochs: int = 100,
              hidden: int = 64, latent: int = 8,
              batch_size: int = 64, lr: float = 1e-3,
              beta: float = 0.1, patience: int = 20,
              device: str = "cpu") -> dict:
    """Train a VAE on normal data only."""
    torch.manual_seed(42)
    n_features = X_train.shape[1]

    # Split 85/15 for early stopping
    split = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:split], X_train[split:]

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr)),
        batch_size=batch_size, shuffle=True,
    )

    model = VAE(n_features=n_features, hidden=hidden, latent=latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        for (batch,) in train_dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            recon, mu, logvar = model(X_val_t)
            val_loss, _, _ = vae_loss(recon, X_val_t, mu, logvar, beta=beta)
            val_loss = val_loss.item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return {"model": model, "best_val": best_val, "n_features": n_features}


def variational_autoencoder(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             feature_cols=None, **kwargs) -> np.ndarray:
    """
    Scorer interface compatible with WindBench baselines.
    Returns severity 0-100.
    """
    if feature_cols is None:
        from src.benchmark.baselines import _get_numeric_features
        feature_cols = _get_numeric_features(train_df)

    # Require at least a few features
    if len(feature_cols) < 3:
        return np.zeros(len(test_df))

    X_train = train_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    scaler = RobustScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Shorter training for benchmark speed (full training would be longer)
    result = train_vae(X_train_s, n_epochs=60, patience=10)
    model = result["model"]

    train_scores = vae_anomaly_score(model, torch.FloatTensor(X_train_s))
    test_scores = vae_anomaly_score(model, torch.FloatTensor(X_test_s))

    # Percentile-rank severity
    sorted_train = np.sort(train_scores)
    ranks = np.searchsorted(sorted_train, test_scores, side="right")
    severity = (ranks / max(len(sorted_train), 1)) * 100.0
    return np.clip(severity, 0.0, 100.0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    df = pd.read_parquet("data/benchmark/harmonized/care_farm_a.parquet")
    # Small sample for demo
    df = df.head(10000)
    split = int(len(df) * 0.6)
    train, test = df.iloc[:split], df.iloc[split:]
    scores = variational_autoencoder(train, test)
    print(f"VAE scored {len(scores)} rows, mean severity {scores.mean():.1f}")
