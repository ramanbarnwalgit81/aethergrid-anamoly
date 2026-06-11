"""
Transformer-Autoencoder Baseline — attention-based anomaly detection.

Matches the Transformer branch of Hybrid Autoencoder 2025 (arxiv 2510.15010).

Architecture:
  Input sequence -> positional encoding -> Transformer encoder (self-attention)
  -> bottleneck -> Transformer decoder -> reconstruction

Attention captures long-range dependencies that LSTM struggles with
(e.g. slow thermal drift over hundreds of timesteps).
"""

from pathlib import Path
import os, sys
import math

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                              -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq, d_model]
        return x + self.pe[:, :x.size(1), :]


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 seq_len: int = 24, dim_ff: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(seq_len, 512))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        # x: [batch, seq, n_features]
        h = self.input_proj(x)
        h = self.pos_enc(h)
        z = self.encoder(h)
        # Bottleneck: project then back to d_model (simple)
        out = self.decoder(z)
        return self.output_proj(out)


def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
    windows = np.empty((n - seq_len + 1, seq_len, X.shape[1]), dtype=np.float32)
    for i in range(n - seq_len + 1):
        windows[i] = X[i:i + seq_len]
    return windows


def train_transformer_ae(X_train: np.ndarray, seq_len: int = 24,
                           n_epochs: int = 30, patience: int = 5,
                           d_model: int = 64, n_heads: int = 4,
                           batch_size: int = 64, lr: float = 1e-3,
                           device: str = "cpu") -> dict:
    torch.manual_seed(42)
    n_features = X_train.shape[1]

    seqs = make_sequences(X_train, seq_len)
    if len(seqs) < 50:
        return {"model": None, "seq_len": seq_len, "n_features": n_features}

    split = int(len(seqs) * 0.85)
    tr, val = seqs[:split], seqs[split:]

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(tr)),
        batch_size=batch_size, shuffle=True,
    )

    model = TransformerAutoencoder(
        n_features=n_features, d_model=d_model, n_heads=n_heads,
        seq_len=seq_len,
    ).to(device)
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
            # Mini-batch the validation forward (see lstm_ae.py) to keep memory
            # bounded on large pooled validation sets (Farm C, 952 features).
            vb = 512
            tot_loss, tot_n = 0.0, 0
            for j in range(0, len(val), vb):
                chunk = torch.FloatTensor(val[j:j + vb]).to(device)
                tot_loss += loss_fn(model(chunk), chunk).item() * len(chunk)
                tot_n += len(chunk)
            val_loss = tot_loss / max(tot_n, 1)

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


def transformer_ae_anomaly_score(bundle: dict, X: np.ndarray) -> np.ndarray:
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
        batch = 128
        for i in range(0, len(seqs), batch):
            chunk = torch.FloatTensor(seqs[i:i + batch])
            recon = model(chunk)
            err = ((recon - chunk) ** 2).mean(dim=(1, 2)).numpy()
            scores_window[i:i + batch] = err

    row_scores = np.empty(n, dtype=np.float32)
    row_scores[:seq_len - 1] = scores_window[0]
    row_scores[seq_len - 1:] = scores_window
    return row_scores
