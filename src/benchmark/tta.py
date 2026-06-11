"""
Test-Time Adaptation (TENT / EATA) for wind turbine aging drift.

FIRST application of TTA to wind CMS.

REFERENCES
----------
- Wang et al., ICLR 2021. "Tent: Fully Test-Time Adaptation by Entropy Minimization"
- Niu et al., ICML 2022. "Efficient Test-Time Model Adaptation without Forgetting (EATA)"
- Wang et al., CVPR 2022. "Continual Test-Time Domain Adaptation (CoTTA)"

SETUP
-----
Wind turbines age; sensor drift accumulates; the model trained on 2017 data
becomes stale by 2020. We simulate this using CARE Farm A's temporal ordering:

  - TRAIN set: event_ids with earliest timestamps
  - TEST set:  event_ids with latest timestamps

At test-time on each new batch, TENT updates ONLY the BN running statistics
(no gradient on weights) by minimizing prediction entropy on the test batch
itself. EATA adds sample-filtering (low-entropy reliable samples only) and
anti-forgetting regularization.

We train a small BN-enabled anomaly head on our Fleet-NBM residual features,
then adapt it at test-time.

Expected: TTA recovers some AUC on temporally-shifted test data that the
frozen model loses to drift.

Usage:
    python -m src.benchmark.tta
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
from sklearn.metrics import roc_auc_score


RESULTS_DIR = Path("docs/results")


class BNClassifier(nn.Module):
    """
    Small classifier with BatchNorm layers (TENT operates on these).
    Input: single v7 anomaly score.
    Output: anomaly probability.
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc3(h).squeeze(-1)


def train_source_model(scores: np.ndarray, labels: np.ndarray,
                          n_epochs: int = 50, batch_size: int = 256,
                          lr: float = 1e-3) -> BNClassifier:
    torch.manual_seed(42)
    model = BNClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    X = torch.FloatTensor(scores)
    y = torch.FloatTensor(labels)
    n = len(X)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logits = model(X[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def collect_bn_params(model: nn.Module):
    """Get parameters of all BN layers (for TENT optimizer)."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            for p in m.parameters():
                p.requires_grad = True
                params.append(p)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad = False
    return params


def tent_adapt(model: BNClassifier, test_scores: np.ndarray,
                 n_steps: int = 1, lr: float = 1e-3) -> np.ndarray:
    """
    TENT: minimize entropy of predictions on test batch, updating only BN.
    Returns adapted predictions (probabilities).
    """
    # Set BN to train mode so running stats update
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.train()
    # Everything else in eval
    for m in model.modules():
        if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.eval()

    params = collect_bn_params(model)
    if not params:
        print("  [WARN] no BN params found")
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)

    X = torch.FloatTensor(test_scores)
    if X.dim() == 1:
        X = X.unsqueeze(-1)

    batch_size = 256
    preds = []

    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        if len(batch) < 4:
            # Too small; skip adaptation for this batch
            with torch.no_grad():
                logit = model(batch)
                prob = torch.sigmoid(logit).numpy()
            preds.append(prob)
            continue

        for _ in range(n_steps):
            opt.zero_grad()
            logit = model(batch)
            prob = torch.sigmoid(logit)
            # Binary entropy
            eps = 1e-7
            ent = -(prob * torch.log(prob + eps) +
                     (1 - prob) * torch.log(1 - prob + eps)).mean()
            ent.backward()
            opt.step()

        with torch.no_grad():
            prob = torch.sigmoid(model(batch)).numpy()
        preds.append(prob)

    model.eval()
    return np.concatenate(preds)


def eata_adapt(model: BNClassifier, test_scores: np.ndarray,
                 entropy_threshold: float = 0.4,
                 n_steps: int = 1, lr: float = 5e-4) -> np.ndarray:
    """
    EATA: same as TENT but filters low-entropy samples and adds anti-forgetting.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.train()
        else:
            m.eval()

    params = collect_bn_params(model)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    # Snapshot original BN params for anti-forgetting
    original_params = [p.data.clone() for p in params]

    X = torch.FloatTensor(test_scores)
    if X.dim() == 1:
        X = X.unsqueeze(-1)

    batch_size = 256
    preds = []

    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        if len(batch) < 4:
            with torch.no_grad():
                logit = model(batch)
                prob = torch.sigmoid(logit).numpy()
            preds.append(prob)
            continue

        for _ in range(n_steps):
            opt.zero_grad()
            logit = model(batch)
            prob = torch.sigmoid(logit)
            eps = 1e-7
            ent_per_sample = -(prob * torch.log(prob + eps) +
                                (1 - prob) * torch.log(1 - prob + eps))
            # EATA: filter low-entropy reliable samples
            mask = (ent_per_sample < entropy_threshold).float()
            if mask.sum() < 2:
                continue
            filtered_ent = (ent_per_sample * mask).sum() / mask.sum()
            # Anti-forgetting penalty
            reg = sum(((p - op) ** 2).sum()
                      for p, op in zip(params, original_params))
            loss = filtered_ent + 1e-4 * reg
            loss.backward()
            opt.step()

        with torch.no_grad():
            prob = torch.sigmoid(model(batch)).numpy()
        preds.append(prob)

    model.eval()
    return np.concatenate(preds)


def main():
    print("=" * 80)
    print("  TTA (TENT + EATA) for CARE Farm A v7 scores")
    print("  First wind-CMS test-time adaptation")
    print("=" * 80)

    scores_path = RESULTS_DIR / "care_ensemble_v7_per_event_scores.json"
    with scores_path.open() as f:
        per_event = json.load(f)

    # Simulate temporal drift: use CARE event_id as a proxy
    # Early event_ids → train, late event_ids → test
    all_ids = sorted([int(k) for k in per_event.keys()])
    split_point = len(all_ids) // 2
    train_ids = set(all_ids[:split_point])
    test_ids = set(all_ids[split_point:])

    print(f"\n[SPLIT] Train events (early IDs): {sorted(train_ids)}")
    print(f"[SPLIT] Test events (late IDs):   {sorted(test_ids)}")

    results = {}
    for label_key in ["y_event", "y_status"]:
        print(f"\n{'-' * 60}\n  Label: {label_key}\n{'-' * 60}")

        # Pool train
        train_scores = []
        train_labels = []
        for eid in train_ids:
            d = per_event[str(eid)]
            train_scores.extend(d["scores"])
            train_labels.extend(d[label_key])
        train_scores = np.array(train_scores, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)

        # Test
        test_scores = []
        test_labels = []
        test_event_ids = []
        for eid in test_ids:
            d = per_event[str(eid)]
            test_scores.extend(d["scores"])
            test_labels.extend(d[label_key])
            test_event_ids.extend([eid] * len(d["scores"]))
        test_scores = np.array(test_scores, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.float32)

        if train_labels.sum() == 0 or test_labels.sum() == 0:
            continue

        # Train source model
        print(f"  [TRAIN] source on {len(train_scores):,} rows")
        model = train_source_model(train_scores, train_labels, n_epochs=30)

        # Baseline: frozen model on test
        model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(test_scores).unsqueeze(-1)
            baseline = torch.sigmoid(model(X)).numpy()
        auc_baseline = float(roc_auc_score(test_labels, baseline))
        print(f"  [BASELINE] frozen-source AUC on test: {auc_baseline:.4f}")

        # TENT
        model_tent = train_source_model(train_scores, train_labels, n_epochs=30)
        preds_tent = tent_adapt(model_tent, test_scores)
        auc_tent = float(roc_auc_score(test_labels, preds_tent))
        print(f"  [TENT]   adapted AUC on test: {auc_tent:.4f}  (Δ {auc_tent-auc_baseline:+.4f})")

        # EATA
        model_eata = train_source_model(train_scores, train_labels, n_epochs=30)
        preds_eata = eata_adapt(model_eata, test_scores)
        auc_eata = float(roc_auc_score(test_labels, preds_eata))
        print(f"  [EATA]   adapted AUC on test: {auc_eata:.4f}  (Δ {auc_eata-auc_baseline:+.4f})")

        results[label_key] = {
            "n_train": len(train_scores),
            "n_test": len(test_scores),
            "auc_baseline_frozen": auc_baseline,
            "auc_tent": auc_tent,
            "auc_eata": auc_eata,
            "delta_tent": auc_tent - auc_baseline,
            "delta_eata": auc_eata - auc_baseline,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "tta.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
