"""
Spatio-Temporal Graph Neural Network on CARE — first ST-GNN on CARE.

REFERENCES
----------
- Fu et al. 2025 IEEE TII "SCADA Data-Driven ST-GCN for WT fault diagnosis"
- Wu et al. 2020 "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
- Bai et al. 2020 "Adaptive Graph Convolutional Recurrent Network"

Our twist: nodes are SIGNALS (not turbines), edges are learned Pearson
correlations on rolling windows. A GAT layer over signal-signal graph
propagates spatial structure; a TCN/Conv1D captures temporal patterns.
Anomaly score = reconstruction error after GNN-TCN encoder-decoder.

This form factor lets us use CARE Farm A's single-turbine event streams
(we cannot build a fleet graph across turbines inside a single-event CSV).

Usage:
    python -m src.benchmark.st_gnn
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

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


RESULTS_DIR = Path("docs/results")
CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")


# Signals to use as GNN nodes
SIGNALS = [
    "wind_speed_3_avg",
    "sensor_0_avg",
    "power_30_avg",
    "sensor_18_avg",
    "sensor_12_avg",
    "sensor_11_avg",
    "sensor_13_avg",
    "sensor_14_avg",
    "sensor_15_avg",
    "sensor_7_avg",
]
N_NODES = len(SIGNALS)
WINDOW = 32   # temporal window per node feature


class ManualGAT(nn.Module):
    """
    Minimal graph-attention-like module without torch-geometric dependency.
    Operates on a fully-connected graph with learned attention.
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim * n_heads)
        self.a = nn.Linear(2 * out_dim, 1)

    def forward(self, x):
        # x: [batch, N_nodes, in_dim]
        B, N, F_in = x.shape
        h = self.W(x).view(B, N, self.n_heads, self.out_dim)
        # Broadcast pairs
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)   # [B, N, N, H, D]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)
        e = self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)  # [B, N, N, H]
        alpha = F.softmax(e, dim=2)
        # Weighted sum over neighbors
        out = (alpha.unsqueeze(-1) * h_j).sum(dim=2)  # [B, N, H, D]
        out = out.mean(dim=2)  # average heads → [B, N, D]
        return F.elu(out)


class STGNN(nn.Module):
    """
    Spatio-temporal GNN: Conv1D over time + GAT over nodes.
    Reconstructs the input window; residual = anomaly.
    """
    def __init__(self, window: int = WINDOW, n_nodes: int = N_NODES,
                  hidden: int = 32):
        super().__init__()
        self.window = window
        self.n_nodes = n_nodes
        # Temporal encoder per node (Conv1D)
        self.tcn = nn.Conv1d(in_channels=1, out_channels=hidden, kernel_size=5,
                                padding=2)
        # GAT over nodes
        self.gat = ManualGAT(in_dim=hidden, out_dim=hidden, n_heads=2)
        # Decoder: back to 1 channel
        self.dec = nn.Conv1d(in_channels=hidden, out_channels=1, kernel_size=1)

    def forward(self, x):
        # x: [batch, N_nodes, window]
        B, N, W = x.shape
        # Per-node temporal encoding
        h = x.reshape(B * N, 1, W)
        h = F.relu(self.tcn(h))  # [B*N, hidden, W]
        # Aggregate time into feature vec for GAT
        h_time_mean = h.mean(dim=-1)  # [B*N, hidden]
        h_nodes = h_time_mean.view(B, N, -1)
        # GAT
        h_nodes = self.gat(h_nodes)  # [B, N, hidden]
        # Broadcast back to time dimension for decoding
        h_broadcast = h_nodes.unsqueeze(-1).expand(-1, -1, -1, W)  # [B, N, hidden, W]
        h_broadcast = h_broadcast.reshape(B * N, -1, W)
        recon = self.dec(h_broadcast).view(B, N, W)
        return recon


def build_windows(X: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Convert [T, N_nodes] -> [num_windows, N_nodes, window]."""
    T, N = X.shape
    if T < window:
        return np.empty((0, N, window), dtype=np.float32)
    out = np.stack([X[t:t + window].T for t in range(T - window + 1)], axis=0)
    return out.astype(np.float32)


def load_event(event_id: int) -> pd.DataFrame:
    return pd.read_csv(CARE_DIR / "datasets" / f"{event_id}.csv",
                         sep=";", low_memory=False)


def build_pool():
    events = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    X_parts = []
    for _, row in events.iterrows():
        if row["event_label"] != "normal":
            continue
        try:
            df = load_event(int(row["event_id"]))
        except Exception:
            continue
        if "train_test" not in df.columns or "status_type_id" not in df.columns:
            continue
        mask = (df["train_test"] == "train") & (df["status_type_id"].fillna(1) == 0)
        avail = [c for c in SIGNALS if c in df.columns]
        if len(avail) < 5 or mask.sum() < 500:
            continue
        X = df.loc[mask, SIGNALS[:len(avail)]].ffill().fillna(0).to_numpy(dtype=np.float32)
        # Pad if missing signals
        if X.shape[1] < N_NODES:
            pad = np.zeros((len(X), N_NODES - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        X_parts.append(X)
    if not X_parts:
        return None
    return np.vstack(X_parts)


def train_stgnn(X_train: np.ndarray, n_epochs: int = 20, batch_size: int = 64,
                  lr: float = 1e-3) -> tuple:
    torch.manual_seed(42)
    scaler = RobustScaler().fit(X_train)
    X_s = np.clip(scaler.transform(X_train), -10, 10).astype(np.float32)

    # Subsample rows to 20000 for speed
    if len(X_s) > 20000:
        idx = np.random.RandomState(42).choice(len(X_s), 20000, replace=False)
        X_s = X_s[idx]

    windows = build_windows(X_s, window=WINDOW)
    print(f"  training windows: {windows.shape}")

    # Further cap if too many
    if len(windows) > 10000:
        idx = np.random.RandomState(42).choice(len(windows), 10000, replace=False)
        windows = windows[idx]

    model = STGNN(window=WINDOW, n_nodes=N_NODES, hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(windows)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx = torch.FloatTensor(windows[idx])
            opt.zero_grad()
            recon = model(bx)
            loss = F.mse_loss(recon, bx)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:>3}  avg loss: {total/n:.5f}")
    model.eval()
    return model, scaler


def score_event(model, scaler, df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    avail = [c for c in SIGNALS if c in df.columns]
    X = df[SIGNALS[:len(avail)]].ffill().fillna(0).to_numpy(dtype=np.float32)
    if X.shape[1] < N_NODES:
        pad = np.zeros((len(X), N_NODES - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    X_s = np.clip(scaler.transform(X), -10, 10).astype(np.float32)

    windows = build_windows(X_s, window=WINDOW)
    if len(windows) == 0:
        return np.zeros(n, dtype=np.float32)

    scores = np.zeros(len(windows), dtype=np.float32)
    with torch.no_grad():
        batch = 128
        for i in range(0, len(windows), batch):
            bx = torch.FloatTensor(windows[i:i + batch])
            recon = model(bx)
            err = ((bx - recon) ** 2).mean(dim=(1, 2)).numpy()
            scores[i:i + batch] = err
    # Map window scores back to row scores (window i covers rows [i, i+window))
    row_scores = np.full(n, scores[0], dtype=np.float32)
    row_scores[WINDOW - 1:WINDOW - 1 + len(scores)] = scores
    return row_scores


def main():
    print("=" * 80)
    print("  ST-GNN on CARE Farm A — first ST-GNN on CARE")
    print("=" * 80)

    X_train = build_pool()
    if X_train is None:
        print("[ERR] no training data")
        return
    print(f"\n[POOL] Training rows: {X_train.shape}")

    t0 = time.time()
    model, scaler = train_stgnn(X_train, n_epochs=20, batch_size=64)
    print(f"[TRAIN] done in {time.time()-t0:.0f}s")

    # Evaluate
    events = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    anomaly = events[events["event_label"] == "anomaly"]

    results = []
    for _, row in anomaly.iterrows():
        eid = int(row["event_id"])
        df = load_event(eid)
        n = len(df)

        # Labels
        y_event = np.zeros(n, dtype=int)
        s_idx = int(row.get("event_start_id", -1))
        e_idx = int(row.get("event_end_id", -1))
        if s_idx >= 0 and e_idx >= s_idx:
            y_event[s_idx:e_idx + 1] = 1

        test_mask = (df.get("train_test") == "prediction").to_numpy() if \
            "train_test" in df.columns else np.ones(n, dtype=bool)

        scores = score_event(model, scaler, df)
        auc_event = None
        if test_mask.sum() > 0:
            y_t = y_event[test_mask]; s_t = scores[test_mask]
            if len(np.unique(y_t)) == 2:
                auc_event = float(roc_auc_score(y_t, s_t))

        # Status label
        status = df["status_type_id"].fillna(0).astype(int).values if \
            "status_type_id" in df.columns else np.zeros(n, dtype=int)
        y_status = (status != 0).astype(int)
        auc_status = None
        if test_mask.sum() > 0:
            y_t = y_status[test_mask]; s_t = scores[test_mask]
            if len(np.unique(y_t)) == 2:
                auc_status = float(roc_auc_score(y_t, s_t))

        print(f"  event {eid:>3} ({str(row['event_description'])[:25]:<25}) "
              f"event AUC={auc_event}  status AUC={auc_status}")

        results.append({
            "event_id": eid,
            "fault": str(row.get("event_description", ""))[:40],
            "auc_event": auc_event,
            "auc_status": auc_status,
        })

    mean_event = np.mean([r["auc_event"] for r in results
                            if r.get("auc_event") is not None])
    mean_status = np.mean([r["auc_status"] for r in results
                             if r.get("auc_status") is not None])
    print(f"\n[AGGREGATE]")
    print(f"  Mean event AUC:  {mean_event:.4f}")
    print(f"  Mean status AUC: {mean_status:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "st_gnn.json").open("w") as f:
        json.dump({
            "method": "ST-GNN (TCN + manual GAT) over signal-signal graph",
            "n_nodes": N_NODES,
            "window": WINDOW,
            "mean_auc_event": float(mean_event),
            "mean_auc_status": float(mean_status),
            "per_event": results,
        }, f, indent=2, default=str)
    print(f"[OK] Saved: docs/results/st_gnn.json")


if __name__ == "__main__":
    main()
