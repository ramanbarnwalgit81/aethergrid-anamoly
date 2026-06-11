"""
LSTM-Autoencoder Baseline — captures temporal dependencies in SCADA sequences.

Matches the LSTM-AE branch of Hybrid Autoencoder 2025 (arxiv 2510.15010).

Architecture:
  Encoder: LSTM(input_dim -> hidden) -> latent
  Decoder: latent -> LSTM(hidden -> input_dim)

Anomaly score: per-step reconstruction MSE (mean over window).

Captures temporal signatures a static VAE misses:
  - Drift trends (oil temp creeping up)
  - Oscillation patterns (bearing wear harmonics)
  - Sequence-level breakdowns in coupled sensors
"""

from pathlib import Path
import os, sys

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder."""

    def __init__(self, n_features: int, hidden: int = 64,
                 n_layers: int = 1, seq_len: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_features = n_features

        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
        )
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        _, (h, c) = self.encoder(x)
        # Repeat the final hidden state seq_len times as decoder input
        repeated = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Convert [n_samples, n_features] -> [n_windows, seq_len, n_features]."""
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
    windows = np.empty((n - seq_len + 1, seq_len, X.shape[1]), dtype=np.float32)
    for i in range(n - seq_len + 1):
        windows[i] = X[i:i + seq_len]
    return windows


def train_lstm_ae(X_train: np.ndarray, seq_len: int = 24,
                   n_epochs: int = 30, patience: int = 5,
                   hidden: int = 64, batch_size: int = 64,
                   lr: float = 1e-3, device: str = "cpu") -> dict:
    """Train LSTM-AE on normal data windows."""
    torch.manual_seed(42)
    n_features = X_train.shape[1]

    seqs = make_sequences(X_train, seq_len)
    if len(seqs) < 50:
        return {"model": None, "seq_len": seq_len, "n_features": n_features}

    split = int(len(seqs) * 0.85)
    tr = seqs[:split]
    val = seqs[split:]

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(tr)),
        batch_size=batch_size, shuffle=True,
    )

    model = LSTMAutoencoder(n_features=n_features, hidden=hidden,
                             seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        for (batch,) in train_dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_t = torch.FloatTensor(val).to(device)
            val_loss = loss_fn(model(val_t), val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "seq_len": seq_len, "n_features": n_features,
            "best_val": best_val}


def lstm_ae_anomaly_score(bundle: dict, X: np.ndarray) -> np.ndarray:
    """
    Return per-row anomaly score. Uses last-step reconstruction error of the
    window ending at each row; first (seq_len - 1) rows inherit the first score.
    """
    model = bundle.get("model")
    seq_len = bundle["seq_len"]
    n = len(X)
    if model is None:
        return np.zeros(n, dtype=np.float32)

    seqs = make_sequences(X, seq_len)
    if len(seqs) == 0:
        return np.zeros(n, dtype=np.float32)

    model.eval()
    scores_window = np.empty(len(seqs), dtype=np.float32)
    with torch.no_grad():
        # Batch through to avoid OOM
        batch = 256
        for i in range(0, len(seqs), batch):
            chunk = torch.FloatTensor(seqs[i:i + batch])
            recon = model(chunk)
            # Per-window MSE: mean over seq_len and features
            err = ((recon - chunk) ** 2).mean(dim=(1, 2)).numpy()
            scores_window[i:i + batch] = err

    # Map window scores to per-row scores — window i covers rows [i, i+seq_len)
    # Anomaly at row t is represented by window starting at t - seq_len + 1
    # We'll assign to row (i + seq_len - 1) the score of window i
    row_scores = np.empty(n, dtype=np.float32)
    row_scores[:seq_len - 1] = scores_window[0]  # pad start
    row_scores[seq_len - 1:] = scores_window
    return row_scores
