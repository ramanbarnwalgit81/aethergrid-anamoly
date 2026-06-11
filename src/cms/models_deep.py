"""
Deep normal-behavior autoencoders (PyTorch, CPU-friendly).

Same contract as src.cms.models scorers:  fit(X_train) ; score(X) -> higher=worse.

  DenseAE  — denoising symmetric autoencoder; reconstruction MSE is the score.
             This is the architecture class behind the published CARE 0.66 baseline.
  VAE      — variational AE; score = reconstruction error + KL (negative ELBO),
             often more robust to nuisance variation in normal operation.

Determinism: fixed seeds; input is already standardized by the Preprocessor.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


def _seed(s: int = 42):
    torch.manual_seed(s)
    np.random.seed(s)


def _bottleneck(d: int) -> tuple[int, int]:
    """Hidden / latent sizes scaled to the input dimensionality."""
    h = max(32, min(256, d // 2))
    z = max(8, min(64, d // 8))
    return h, z


class _DenseAEModule(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        h, z = _bottleneck(d)
        self.enc = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class DenseAEScorer:
    def __init__(self, epochs: int = 40, batch: int = 256, lr: float = 1e-3,
                 noise: float = 0.1, max_train: int = 40000, seed: int = 42):
        if not _TORCH:
            raise ImportError("PyTorch not available")
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.noise, self.max_train, self.seed = noise, max_train, seed
        self.model: _DenseAEModule | None = None

    def fit(self, X: np.ndarray) -> "DenseAEScorer":
        _seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        if len(X) > self.max_train:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False)
            X = X[idx]
        d = X.shape[1]
        self.model = _DenseAEModule(d)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        Xt = torch.from_numpy(X)
        n = len(Xt)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                b = Xt[perm[i:i + self.batch]]
                inp = b + self.noise * torch.randn_like(b) if self.noise else b
                opt.zero_grad()
                loss = loss_fn(self.model(inp), b)
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad() if _TORCH else (lambda f: f)
    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = self.model(torch.from_numpy(X)).numpy()
        return np.sqrt(np.mean((X - out) ** 2, axis=1))


class _VAEModule(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        h, z = _bottleneck(d)
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.mu = nn.Linear(h, z)
        self.logvar = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        he = self.enc(x)
        mu, logvar = self.mu(he), self.logvar(he)
        std = torch.exp(0.5 * logvar)
        zs = mu + std * torch.randn_like(std)
        return self.dec(zs), mu, logvar


class VAEScorer:
    def __init__(self, epochs: int = 40, batch: int = 256, lr: float = 1e-3,
                 beta: float = 1.0, max_train: int = 40000, seed: int = 42):
        if not _TORCH:
            raise ImportError("PyTorch not available")
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.beta, self.max_train, self.seed = beta, max_train, seed
        self.model: _VAEModule | None = None

    def fit(self, X: np.ndarray) -> "VAEScorer":
        _seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        if len(X) > self.max_train:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False)
            X = X[idx]
        self.model = _VAEModule(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        Xt = torch.from_numpy(X)
        n = len(Xt)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                b = Xt[perm[i:i + self.batch]]
                opt.zero_grad()
                recon, mu, logvar = self.model(b)
                rec = ((recon - b) ** 2).mean()
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                (rec + self.beta * kld).backward()
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad() if _TORCH else (lambda f: f)
    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        recon, mu, logvar = self.model(torch.from_numpy(X))
        recon = recon.numpy()
        rec = np.mean((X - recon) ** 2, axis=1)
        return np.sqrt(rec)


DEEP_SCORERS = {"DenseAE": DenseAEScorer, "VAE": VAEScorer}
