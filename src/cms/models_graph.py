"""
Graph Deviation Network (GDN-style) for wind-turbine SCADA anomaly detection.

Idea (Deng & Hooi, AAAI 2021, adapted): each sensor is a node with a learned
embedding. A sparse directed graph is induced by top-k cosine similarity of the
embeddings — i.e. the model *discovers* which sensors normally drive which. A
graph-attention forecaster predicts each sensor's next value from its learned
neighbours; on normal data the relationships hold, so forecasts are accurate.
When a fault perturbs the inter-sensor physics (e.g. gearbox temp decouples from
load), forecasts break and the per-sensor deviation spikes.

Anomaly score(t) = max over sensors of the median/IQR-normalised forecast error
(GDN's deviation score). This captures *nonlinear inter-sensor* structure that
reconstruction AEs (the published 0.66 baseline) and linear Mahalanobis miss.
The learned adjacency is an interpretable artifact (a paper figure).

CPU-friendly: dense attention over N nodes (N<=~150 after optional PCA on very
high-dim farms), short window, capped training windows, fixed seed.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False

from sklearn.decomposition import PCA


def _seed(s=42):
    torch.manual_seed(s)
    np.random.seed(s)


class _GDN(nn.Module):
    def __init__(self, n_nodes: int, window: int, emb_dim: int = 64,
                 hidden: int = 64, topk: int = 15):
        super().__init__()
        self.n = n_nodes
        self.topk = min(topk, n_nodes - 1)
        self.emb = nn.Embedding(n_nodes, emb_dim)
        # Per-node temporal feature extractor over the input window.
        self.win = nn.Linear(window, hidden)
        # Attention over [h_i || h_j] for neighbour aggregation.
        self.att = nn.Linear(2 * hidden, 1)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    def _adjacency(self):
        e = F.normalize(self.emb.weight, dim=1)        # (N, d)
        sim = e @ e.t()                                 # (N, N) cosine
        sim.fill_diagonal_(-1e9)
        idx = torch.topk(sim, self.topk, dim=1).indices  # (N, topk) neighbours
        return idx

    def forward(self, x):  # x: (B, window, N)
        B = x.shape[0]
        h = self.win(x.permute(0, 2, 1))                # (B, N, hidden)
        nbr = self._adjacency()                          # (N, topk)
        hi = h.unsqueeze(2).expand(B, self.n, self.topk, h.shape[-1])
        hj = h[:, nbr, :]                                # (B, N, topk, hidden)
        a = self.att(torch.cat([hi, hj], dim=-1)).squeeze(-1)   # (B, N, topk)
        a = torch.softmax(F.leaky_relu(a, 0.2), dim=-1)
        agg = (a.unsqueeze(-1) * hj).sum(dim=2)          # (B, N, hidden)
        z = agg * self.emb.weight.unsqueeze(0)           # gate by node identity
        return self.out(z).squeeze(-1)                   # (B, N) next-step forecast


class GDNScorer:
    def __init__(self, window: int = 5, epochs: int = 8, batch: int = 512,
                 lr: float = 2e-3, emb_dim: int = 48, hidden: int = 48,
                 topk: int = 10, reduce_above: int = 50, reduce_to: int = 40,
                 max_windows: int = 4000, seed: int = 42):
        if not _TORCH:
            raise ImportError("PyTorch not available")
        self.window, self.epochs, self.batch, self.lr = window, epochs, batch, lr
        self.emb_dim, self.hidden, self.topk = emb_dim, hidden, topk
        self.reduce_above, self.reduce_to = reduce_above, reduce_to
        self.max_windows, self.seed = max_windows, seed
        self.pca: PCA | None = None
        self.model: _GDN | None = None
        self.err_med = None
        self.err_iqr = None

    def _project(self, X):
        return self.pca.transform(X) if self.pca is not None else X

    def _windows(self, X):
        """Build (M, window, N) inputs and (M, N) next-step targets."""
        w = self.window
        n = len(X)
        if n <= w:
            return None, None
        idx = np.arange(w, n)
        Xin = np.stack([X[i - w:i] for i in idx]).astype(np.float32)  # (M, w, N)
        Xtg = X[idx].astype(np.float32)                                # (M, N)
        return Xin, Xtg

    def fit(self, X: np.ndarray) -> "GDNScorer":
        _seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] > self.reduce_above:
            k = min(self.reduce_to, X.shape[1], max(2, X.shape[0] - 1))
            self.pca = PCA(n_components=k, random_state=self.seed).fit(X)
            X = self.pca.transform(X).astype(np.float32)
        N = X.shape[1]
        Xin, Xtg = self._windows(X)
        if Xin is None:
            raise ValueError("series shorter than window")
        if len(Xin) > self.max_windows:
            sel = np.random.default_rng(self.seed).choice(len(Xin), self.max_windows, replace=False)
            Xin, Xtg = Xin[sel], Xtg[sel]
        self.model = _GDN(N, self.window, self.emb_dim, self.hidden, self.topk)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        Xi, Xt = torch.from_numpy(Xin), torch.from_numpy(Xtg)
        m = len(Xi)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(m)
            for i in range(0, m, self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                pred = self.model(Xi[b])
                loss = F.mse_loss(pred, Xt[b])
                loss.backward()
                opt.step()
        # Per-node error normalisation from TRAIN residuals (GDN deviation).
        self.model.eval()
        with torch.no_grad():
            err = []
            for i in range(0, m, 1024):
                p = self.model(Xi[i:i + 1024])
                err.append((p - Xt[i:i + 1024]).abs().numpy())
            err = np.concatenate(err, axis=0)
        self.err_med = np.median(err, axis=0)
        q75, q25 = np.percentile(err, [75, 25], axis=0)
        self.err_iqr = (q75 - q25) + 1e-6
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(self._project(np.asarray(X, dtype=np.float32)), dtype=np.float32)
        Xin, Xtg = self._windows(X)
        out = np.zeros(len(X))
        if Xin is None:
            return out
        with torch.no_grad():
            preds = []
            Xi = torch.from_numpy(Xin)
            for i in range(0, len(Xi), 1024):
                preds.append(self.model(Xi[i:i + 1024]).numpy())
        pred = np.concatenate(preds, axis=0)
        dev = np.abs(pred - Xtg)                       # (M, N)
        norm_dev = (dev - self.err_med) / self.err_iqr
        s = np.max(norm_dev, axis=1)                    # GDN: max over sensors
        out[self.window:] = s                          # align to time index
        out[:self.window] = s[0] if len(s) else 0.0
        return out


GRAPH_SCORERS = {"GDN": GDNScorer}
