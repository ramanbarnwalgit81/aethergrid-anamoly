"""
Convert selected manuscript tables into publication-quality bar charts.

Values are taken verbatim from the corresponding tables in docs/paper/main.tex
(Kelmarsh detection, adversarial defence, PINN-v2 vs v7 uplift, deployment
economics). Nothing is recomputed or invented here; this is presentation only.
Outputs 300-dpi PDF+PNG to docs/paper/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("docs/paper/figures")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {FIG / name}.pdf/.png")


def fig_kelmarsh():
    models = ["Isolation\nForest", "LSTM-AE", "One-Class\nSVM", "Dense-AE", "IF+LSTM\nens."]
    auc = [0.887, 0.770, 0.931, 0.909, 0.907]
    aucpr = [0.181, 0.071, 0.402, 0.414, 0.228]
    r10 = [0.750, 0.333, 0.792, 0.708, 0.833]
    x = np.arange(len(models)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - w, auc, w, label="AUC-ROC", color="#1f4e79")
    ax.bar(x, aucpr, w, label="AUC-PR", color="#c0504d")
    ax.bar(x + w, r10, w, label="Recall @ FPR=0.10", color="#9bbb59")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("Injected-fault detection on Kelmarsh", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_tbl_kelmarsh")


def fig_adv():
    attacks = ["FGSM\n(eps=0.05)", "PGD\n(eps=0.10)", "Physics-constr.\n(eps=0.10)"]
    no_def = [4.2, 20.8, 12.5]
    with_def = [0.0, 0.0, 0.0]
    x = np.arange(len(attacks)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - w / 2, no_def, w, label="No defence", color="#c0504d")
    ax.bar(x + w / 2, with_def, w, label="With physics-consistency defence", color="#4f81bd")
    for xi, v in zip(x - w / 2, no_def):
        ax.text(xi, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(attacks, fontsize=8)
    ax.set_ylabel("Attack success rate (%)"); ax.set_ylim(0, 24)
    ax.set_title("Adversarial robustness with/without physics defence", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_tbl_adv")


def fig_pinn_v2():
    labels = ["Event-window", "CARE-Precursor", "Status-code"]
    v7 = [0.458, 0.402, 0.636]
    pinn = [0.577, 0.557, 0.646]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    b1 = ax.bar(x - w / 2, v7, w, label="v7 base ensemble", color="#a6a6a6")
    b2 = ax.bar(x + w / 2, pinn, w, label="PINN-v2 stacker", color="#1f4e79")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Mean per-event AUC (Farm A)"); ax.set_ylim(0, 0.75)
    ax.set_title("PINN-v2 vs v7 base across label conventions", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_tbl_pinnv2")


def fig_econ():
    faults = ["Gearbox", "Transformer", "Generator\nbearing", "Hydraulic\ngroup"]
    reactive = [605584, 432560, 175292, 52512]
    predictive = [353780, 252268, 102268, 30756]
    savings = [251804, 180292, 73024, 21756]
    x = np.arange(len(faults)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ax.bar(x - w, np.array(reactive) / 1e3, w, label="Reactive", color="#c0504d")
    ax.bar(x, np.array(predictive) / 1e3, w, label="Predictive", color="#4f81bd")
    ax.bar(x + w, np.array(savings) / 1e3, w, label="Savings", color="#9bbb59")
    ax.set_xticks(x); ax.set_xticklabels(faults, fontsize=8.5)
    ax.set_ylabel("Cost per fault (thousand USD)")
    ax.set_title("Predictive vs reactive maintenance cost (2 MW turbine)", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_tbl_econ")


def main():
    fig_kelmarsh()
    fig_adv()
    fig_pinn_v2()
    fig_econ()


if __name__ == "__main__":
    main()
