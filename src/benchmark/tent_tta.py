"""
TENT Test-Time Adaptation for wind-CMS drift — C3 (Week 2 plan).

First wind-CMS application of entropy-minimization test-time adaptation.

REFERENCE
---------
Wang et al. (ICLR 2021). TENT: Fully Test-time Adaptation by Entropy
Minimization. Original implementation: github.com/DequanWang/tent.

CORE IDEA
---------
At test time (deployment), the model sees turbine data from a new time
period or new operating regime. The frozen-ensemble anomaly score distribution
shifts. Classical approach: retrain. TENT approach: update ONLY batch-norm
statistics (and their affine parameters) via unsupervised entropy minimization.

For our v7 ensemble (VAE + LSTM + Transformer), we TENT-adapt a lightweight
calibration head mapping v7 ensemble scores to probabilities. At test time,
for each new event's prediction region, we tune the head's BN stats on the
test rows themselves using entropy minimization.

Expected (honest): +1 to +5 AUC recovery of drift-induced degradation on
temporal splits. Not a silver bullet, but a legitimate deployment technique.

Usage:
    python -m src.benchmark.tent_tta
"""

from pathlib import Path
import json
import os, sys

_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RESULTS_DIR = Path("docs/results")


def load_v7_scores() -> dict:
    with (RESULTS_DIR / "care_ensemble_v7_per_event_scores.json").open() as f:
        return json.load(f)


class TentableHead(nn.Module):
    """
    Calibration head mapping v7 score → anomaly probability with BN layers
    that TENT can adapt at test time.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc3(h).squeeze(-1)


def train_head(scores: np.ndarray, labels: np.ndarray,
                 n_epochs: int = 80, lr: float = 1e-3) -> TentableHead:
    torch.manual_seed(42)
    model = TentableHead()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    X = torch.FloatTensor(scores).unsqueeze(-1)
    y = torch.FloatTensor(labels)

    model.train()
    for _ in range(n_epochs):
        # Mini-batches
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            idx = perm[i:i + 512]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logits = model(X[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            opt.step()
    return model


def tent_adapt(model: TentableHead, test_scores: np.ndarray,
                 n_iter: int = 10, lr: float = 1e-4) -> TentableHead:
    """
    TENT adaptation: minimize predictive entropy on test rows.
    Only BN parameters are updated. All other weights frozen.
    """
    # Freeze everything except BN affine parameters
    for p in model.parameters():
        p.requires_grad_(False)
    bn_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            bn_params.append(m.weight)
            bn_params.append(m.bias)
            m.track_running_stats = False   # use batch statistics

    model.train()   # BN uses current batch stats (the deployment data)
    opt = torch.optim.Adam(bn_params, lr=lr)

    X = torch.FloatTensor(test_scores).unsqueeze(-1)
    if len(X) < 4:
        return model   # too few rows to adapt meaningfully

    for _ in range(n_iter):
        opt.zero_grad()
        logits = model(X)
        # Entropy on sigmoid probabilities
        p = torch.sigmoid(logits)
        eps = 1e-6
        entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        loss = entropy.mean()
        loss.backward()
        opt.step()

    model.eval()
    return model


def evaluate_with_tta(data: dict, label_key: str, adapt: bool = True) -> dict:
    """
    Leave-one-event-out evaluation. Train calibration head on pooled other
    events, optionally adapt at test time with TENT, measure AUC.
    """
    event_ids = list(data.keys())
    per_event = []

    for held in event_ids:
        train_scores, train_labels = [], []
        for eid, d in data.items():
            if eid == held:
                continue
            train_scores.extend(d["scores"])
            train_labels.extend(d[label_key])
        train_scores = np.array(train_scores, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)

        if train_labels.sum() == 0 or train_labels.sum() == len(train_labels):
            continue

        # Subsample if too large to keep runtime reasonable
        if len(train_scores) > 30000:
            idx = np.random.RandomState(42).choice(len(train_scores), 30000, replace=False)
            train_scores = train_scores[idx]; train_labels = train_labels[idx]

        model = train_head(train_scores, train_labels, n_epochs=60)

        test_scores = np.array(data[held]["scores"], dtype=np.float32)
        test_labels = np.array(data[held][label_key], dtype=np.int64)

        if len(np.unique(test_labels)) < 2:
            continue

        # Baseline (no adaptation)
        model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(test_scores).unsqueeze(-1)
            probs_base = torch.sigmoid(model(X)).numpy()
        auc_base = float(roc_auc_score(test_labels, probs_base))

        # TENT-adapted
        if adapt:
            model_adapted = tent_adapt(model, test_scores, n_iter=10)
            model_adapted.eval()
            with torch.no_grad():
                X = torch.FloatTensor(test_scores).unsqueeze(-1)
                probs_tta = torch.sigmoid(model_adapted(X)).numpy()
            auc_tta = float(roc_auc_score(test_labels, probs_tta))
        else:
            auc_tta = None

        per_event.append({
            "event_id": held,
            "n_test": len(test_scores),
            "auc_baseline": auc_base,
            "auc_tent": auc_tta,
            "delta": (auc_tta - auc_base) if auc_tta is not None else None,
        })

    return per_event


def main():
    print("=" * 80)
    print("  TENT Test-Time Adaptation on v7 scores — first in wind CMS")
    print("=" * 80)

    np.random.seed(42)
    torch.manual_seed(42)
    data = load_v7_scores()

    all_results = {}
    for label in ["y_event", "y_precursor", "y_status"]:
        print(f"\n{'-' * 60}\n  Label: {label}\n{'-' * 60}")
        results = evaluate_with_tta(data, label, adapt=True)
        for r in results:
            delta = r["delta"]
            if delta is not None:
                sign = "+" if delta >= 0 else ""
                print(f"  event {r['event_id']:>4}  n={r['n_test']:>5}  "
                      f"baseline={r['auc_baseline']:.4f}  "
                      f"TENT={r['auc_tent']:.4f}  Δ={sign}{delta:.4f}")

        base_vals = [r["auc_baseline"] for r in results
                     if r["auc_baseline"] is not None]
        tta_vals = [r["auc_tent"] for r in results
                    if r["auc_tent"] is not None]
        deltas = [r["delta"] for r in results if r["delta"] is not None]

        if base_vals and tta_vals:
            print(f"\n  Mean AUC baseline: {np.mean(base_vals):.4f}")
            print(f"  Mean AUC TENT:     {np.mean(tta_vals):.4f}")
            print(f"  Mean Δ:            {np.mean(deltas):+.4f}")
            print(f"  Median Δ:          {np.median(deltas):+.4f}")
            print(f"  Events improved:   {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

        all_results[label] = {
            "per_event": results,
            "mean_auc_baseline": float(np.mean(base_vals)) if base_vals else None,
            "mean_auc_tent": float(np.mean(tta_vals)) if tta_vals else None,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "events_improved_fraction":
                float(sum(1 for d in deltas if d > 0) / max(len(deltas), 1)),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "tent_tta.json").open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Saved: docs/results/tent_tta.json")


if __name__ == "__main__":
    main()
