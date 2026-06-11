"""
Laplace last-layer Bayesian uncertainty on v7 ensemble scores.

Adds epistemic uncertainty decomposition via the Laplace approximation
(Daxberger et al., NeurIPS 2021). Post-hoc — no retraining required.

For each CARE event row, we produce:
  - predictive mean (matches v7 score)
  - aleatoric variance (irreducible noise at this input)
  - epistemic variance (model uncertainty — reducible with more training data)
  - calibration metrics (Expected Calibration Error, reliability diagram data)

COMPLEMENTS C1 (Conformal Risk Control):
  - Conformal: frequentist coverage guarantee on FPR/FNR (distribution-free)
  - Laplace:   Bayesian epistemic uncertainty per point (parametric)
  - Together: the strongest uncertainty story in the wind-CMS literature.

REFERENCE: Daxberger, Kristiadi, Immer, Eschenhagen, Bauer, Hennig (NeurIPS 2021)
"Laplace Redux — Effortless Bayesian Deep Learning"

Usage:
    python -m src.benchmark.laplace_uq
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
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader


RESULTS_DIR = Path("docs/results")


def load_v7_scores() -> dict:
    with (RESULTS_DIR / "care_ensemble_v7_per_event_scores.json").open() as f:
        return json.load(f)


def load_event_info() -> dict:
    care_info = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A/event_info.csv")
    df = pd.read_csv(care_info, sep=";")
    return {int(r["event_id"]): str(r.get("event_description", "normal"))
            for _, r in df.iterrows()}


class CalibrationHead(nn.Module):
    """
    Small feed-forward head that maps v7 anomaly scores to probability.
    We fit Laplace approximation over the LAST LAYER of this head.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),   # last layer — Laplace operates here
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(scores: np.ndarray, labels: np.ndarray,
                 n_epochs: int = 100, lr: float = 1e-3) -> CalibrationHead:
    torch.manual_seed(42)
    model = CalibrationHead()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    X = torch.FloatTensor(scores).unsqueeze(-1)
    y = torch.FloatTensor(labels)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=512, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for bx, by in dl:
            opt.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
    model.eval()
    return model


class ManualLastLayerLaplace:
    """
    Minimal manual implementation of last-layer Laplace approximation.
    Avoids laplace-torch dependency (broken import on Windows venv).

    For a frozen feature extractor f(x) and a linear last layer w^T f(x) + b,
    the Laplace posterior over w given MAP w_map and training (X, y) is:

        p(w | D) ≈ N(w_map, Σ)
        Σ⁻¹ = (1/σ²) Φ^T Φ + (1/τ²) I

    where Φ = f(X) (N × d features from second-to-last layer), σ² is
    observation noise variance (estimated from MAP residuals), and τ² is
    prior variance (optimized via marginal likelihood).

    Predictive variance at new x*:
        Var[y*] = σ² + f(x*)^T Σ f(x*)     (total)
                = σ²  (aleatoric) + f(x*)^T Σ f(x*) (epistemic)
    """

    def __init__(self, model: CalibrationHead):
        self.model = model
        # Last layer is net[-1] (Linear)
        self.last_linear = model.net[-1]
        # Extract features from second-to-last output (pre-final-linear)
        self.feature_extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.w_map = None
        self.b_map = None
        self.sigma2 = None
        self.tau2 = None
        self.sigma_posterior_inv = None

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()
        with torch.no_grad():
            return self.feature_extractor(x)

    def fit(self, calib_scores: np.ndarray, calib_labels: np.ndarray):
        X = torch.FloatTensor(calib_scores).unsqueeze(-1)
        y = torch.FloatTensor(calib_labels)

        # Get features + MAP predictions
        Phi = self._features(X).detach().numpy().astype(np.float64)
        # Add bias column
        Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))])

        # MAP weights from the last linear layer
        w = self.last_linear.weight.detach().cpu().numpy().astype(np.float64).ravel()
        b = float(self.last_linear.bias.detach().cpu().numpy())
        w_full = np.concatenate([w, [b]])
        self.w_map = w_full

        # Predictions
        y_pred = Phi @ w_full
        residuals = y.numpy() - 1.0 / (1.0 + np.exp(-y_pred))   # sigmoid for binary
        # Observation noise: MSE of residuals
        self.sigma2 = float(np.mean(residuals ** 2) + 1e-4)

        # Prior precision: optimize via marginal likelihood grid
        best_mll = -np.inf
        best_tau2 = 1.0
        d = Phi.shape[1]
        for log_tau2 in np.linspace(-3, 3, 25):
            tau2 = float(np.exp(log_tau2))
            # Posterior precision: (1/σ²) Φᵀ Φ + (1/τ²) I
            prec = (1.0 / self.sigma2) * (Phi.T @ Phi) + (1.0 / tau2) * np.eye(d)
            try:
                log_det = np.linalg.slogdet(prec)[1]
                # Simplified marginal log-likelihood (regression approximation)
                mll = -0.5 * log_det - 0.5 * d * np.log(tau2) - 0.5 * (
                    np.sum(residuals ** 2) / self.sigma2
                )
                if mll > best_mll:
                    best_mll = mll
                    best_tau2 = tau2
            except np.linalg.LinAlgError:
                continue
        self.tau2 = best_tau2
        # Final posterior precision
        self.sigma_posterior_inv = ((1.0 / self.sigma2) * (Phi.T @ Phi)
                                     + (1.0 / self.tau2) * np.eye(d))
        return self

    def predict_with_var(self, scores: np.ndarray) -> tuple:
        X = torch.FloatTensor(scores).unsqueeze(-1)
        Phi = self._features(X).detach().numpy().astype(np.float64)
        Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))])

        # Mean prediction via MAP
        mean_logit = Phi @ self.w_map
        mean_prob = 1.0 / (1.0 + np.exp(-mean_logit))

        # Epistemic variance: f(x)^T Σ f(x)
        try:
            sigma_post = np.linalg.inv(self.sigma_posterior_inv)
        except np.linalg.LinAlgError:
            sigma_post = np.linalg.pinv(self.sigma_posterior_inv)
        epistemic_var = np.einsum("ij,jk,ik->i", Phi, sigma_post, Phi)
        epistemic_var = np.maximum(epistemic_var, 0)
        # Aleatoric variance: sigma^2 on logit
        aleatoric_var = np.full_like(epistemic_var, self.sigma2)
        total_var = epistemic_var + aleatoric_var
        return mean_prob, epistemic_var, aleatoric_var, total_var


def fit_laplace(model: CalibrationHead, calib_scores: np.ndarray,
                 calib_labels: np.ndarray, subset: str = "last_layer"):
    """Manual last-layer Laplace."""
    la = ManualLastLayerLaplace(model)
    la.fit(calib_scores, calib_labels)
    return la


def predict_with_uncertainty(la, scores: np.ndarray) -> tuple:
    mean_prob, epistemic_var, aleatoric_var, total_var = la.predict_with_var(scores)
    return mean_prob, total_var, epistemic_var, aleatoric_var


def expected_calibration_error(preds: np.ndarray, labels: np.ndarray,
                                  n_bins: int = 10) -> float:
    """Compute ECE on binary calibration."""
    preds = np.clip(preds, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(preds)
    for i in range(n_bins):
        mask = (preds >= bins[i]) & (preds < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = preds[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram_bins(preds: np.ndarray, labels: np.ndarray,
                                 n_bins: int = 10) -> list:
    preds = np.clip(preds, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (preds >= bins[i]) & (preds < bins[i + 1])
        if mask.sum() == 0:
            continue
        out.append({
            "bin_center": float((bins[i] + bins[i + 1]) / 2),
            "n_samples": int(mask.sum()),
            "mean_confidence": float(preds[mask].mean()),
            "accuracy": float(labels[mask].mean()),
        })
    return out


def main():
    print("=" * 80)
    print("  LAPLACE LAST-LAYER Bayesian UQ on v7 scores")
    print("=" * 80)

    per_event = load_v7_scores()
    fault_map = load_event_info()

    for label_key in ["y_event", "y_status"]:
        print(f"\n{'-' * 60}\n  Label: {label_key}\n{'-' * 60}")

        # Pool ALL events as calibration set (75/25 split within pool)
        all_scores, all_labels = [], []
        for eid, d in per_event.items():
            all_scores.extend(d["scores"])
            all_labels.extend(d[label_key])
        all_scores = np.array(all_scores, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        mask = np.isfinite(all_scores)
        all_scores = all_scores[mask]; all_labels = all_labels[mask]

        if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
            continue

        n = len(all_scores)
        split = int(0.75 * n)
        perm = np.random.RandomState(42).permutation(n)
        train_idx, test_idx = perm[:split], perm[split:]

        train_scores, train_labels = all_scores[train_idx], all_labels[train_idx]
        test_scores, test_labels = all_scores[test_idx], all_labels[test_idx]

        # Train a simple head mapping v7 score → anomaly probability
        print(f"  [TRAIN] Calibration head on {len(train_scores):,} rows...")
        model = train_head(train_scores, train_labels.astype(np.float32))

        # Sigmoid baseline predictions (no Laplace)
        with torch.no_grad():
            x_test = torch.FloatTensor(test_scores).unsqueeze(-1)
            sigmoid_preds = torch.sigmoid(model(x_test)).numpy()

        ece_baseline = expected_calibration_error(sigmoid_preds, test_labels.astype(float))
        auc_baseline = float(roc_auc_score(test_labels, sigmoid_preds))

        # Fit Laplace
        print(f"  [LAPLACE] Fitting last-layer Laplace...")
        try:
            la = fit_laplace(model, train_scores, train_labels.astype(np.float32))
            mean, var, epistemic_var, aleatoric_var = predict_with_uncertainty(la, test_scores)
            epistemic_std = np.sqrt(np.clip(epistemic_var, 0, None))
            laplace_probs = np.clip(mean, 0, 1)

            ece_laplace = expected_calibration_error(laplace_probs, test_labels.astype(float))
            auc_laplace = float(roc_auc_score(test_labels, laplace_probs)) \
                if len(np.unique(test_labels)) == 2 else None
            rel = reliability_diagram_bins(laplace_probs, test_labels.astype(float))

            print(f"  Baseline (sigmoid only):")
            print(f"    AUC = {auc_baseline:.4f}    ECE = {ece_baseline:.4f}")
            print(f"  Laplace last-layer:")
            print(f"    AUC = {auc_laplace:.4f}    ECE = {ece_laplace:.4f}")
            print(f"    Mean epistemic std: {epistemic_std.mean():.4f}")
            print(f"    Epistemic std on anomalies: {epistemic_std[test_labels==1].mean():.4f}")
            print(f"    Epistemic std on normals:   {epistemic_std[test_labels==0].mean():.4f}")

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out = {
                "label": label_key,
                "n_train": len(train_scores),
                "n_test": len(test_scores),
                "auc_baseline_sigmoid": auc_baseline,
                "auc_laplace": auc_laplace,
                "ece_baseline_sigmoid": ece_baseline,
                "ece_laplace": ece_laplace,
                "epistemic_std_mean": float(epistemic_std.mean()),
                "epistemic_std_on_anomalies": float(epistemic_std[test_labels==1].mean()),
                "epistemic_std_on_normals": float(epistemic_std[test_labels==0].mean()),
                "reliability_diagram": rel,
            }
            with (RESULTS_DIR / f"laplace_uq_{label_key}.json").open("w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"  [OK] Saved: docs/results/laplace_uq_{label_key}.json")
        except Exception as e:
            print(f"  [ERR] Laplace fitting failed: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
