"""Generate three figures for the paper from committed JSON artifacts.

Fig. 1 (fig:auc_scatter):  Per-event AUC scatter, PINN-v2 vs v7 base, coloured
                           by fault type. Illustrates where the stacker wins
                           and loses.
Fig. 2 (fig:reliability):  Calibration reliability diagram on event-window and
                           status tasks, sigmoid vs Laplace. Visualises the
                           ECE numbers in Section V-E.
Fig. 3 (fig:event_40):     Event-40 generator-bearing timeline: v7 score,
                           PINN-v2 score, status_type_id overlay, event window
                           marker. Illustrates how PINN-v2 flips the inverse-
                           ranking on this event.

Outputs go to docs/paper/figures/ as PDF (IEEE-preferred) + PNG (for preview).
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("docs/results")
FIG_DIR = Path("docs/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FAULT_COLOURS = {
    "Generator bearing failure": "#d62728",  # red
    "Gearbox failure":            "#1f77b4",  # blue
    "Hydraulic group":            "#2ca02c",  # green
    "Transformer failure":        "#ff7f0e",  # orange
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: Per-event AUC scatter, PINN-v2 vs v7 base
# ─────────────────────────────────────────────────────────────────────────

def fig1_per_event_scatter():
    with (RESULTS_DIR / "pinn_stacker_y_event.json").open() as f:
        d = json.load(f)
    per = d["per_event"]
    fig, ax = plt.subplots(figsize=(3.5, 3.4))

    # Group by fault type for legend
    seen = set()
    for r in per:
        ft = r["fault_type"]
        c = FAULT_COLOURS.get(ft, "#7f7f7f")
        lbl = ft if ft not in seen else None
        seen.add(ft)
        ax.scatter(r["base_v7_auc"], r["pinn_stacker_auc"],
                    color=c, s=45, label=lbl, edgecolors="black", linewidths=0.6,
                    zorder=3)

    # Diagonal (no change)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.7, zorder=1)
    # Chance lines
    ax.axhline(0.5, color="gray", lw=0.5, alpha=0.5, zorder=1)
    ax.axvline(0.5, color="gray", lw=0.5, alpha=0.5, zorder=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("v7 base AUC")
    ax.set_ylabel("PINN-v2 stacker AUC")
    ax.set_title("Per-event AUC: PINN-v2 vs v7 on CARE Farm A\n"
                  "(event-window label; points above diagonal = stacker wins)")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.savefig(FIG_DIR / "fig1_auc_scatter.pdf")
    fig.savefig(FIG_DIR / "fig1_auc_scatter.png")
    plt.close(fig)
    print("  [OK] fig1_auc_scatter.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Reliability diagram (sigmoid vs Laplace, event-window + status)
# ─────────────────────────────────────────────────────────────────────────

def fig2_reliability():
    # Plot the ACTUAL stored per-bin calibration data from the Laplace UQ
    # artifacts (reliability_diagram: mean_confidence vs observed accuracy per
    # bin). The sigmoid and Laplace posteriors have identical AUC and ECE on
    # this pool (ece_baseline_sigmoid == ece_laplace to 6 d.p.), so the stored
    # diagram represents both; we report both ECE values in the legend to make
    # the coincidence explicit rather than drawing one series twice with
    # different markers.
    with (RESULTS_DIR / "laplace_uq_y_event.json").open() as f:
        lap_event = json.load(f)
    with (RESULTS_DIR / "laplace_uq_y_status.json").open() as f:
        lap_status = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    for ax, (lap, task) in zip(
        axes,
        [(lap_event, "Event-window"), (lap_status, "status_type_id")],
    ):
        rd = lap["reliability_diagram"]
        conf = np.array([b["mean_confidence"] for b in rd])
        acc = np.array([b["accuracy"] for b in rd])
        ece_sig = lap["ece_baseline_sigmoid"]
        ece_lap = lap["ece_laplace"]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="perfect")
        ax.plot(conf, acc, "o-", color="#1f77b4", lw=1.5, ms=5,
                label=f"sigmoid = Laplace\n(ECE {ece_sig:.3f} vs {ece_lap:.3f})")
        # small margin so markers at the 0/1 boundaries are not clipped by the spine
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
        ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("mean predicted probability")
        ax.set_title(f"{task} label")
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(loc="upper left", frameon=True, fontsize=7.5)
    axes[0].set_ylabel("observed anomaly rate")
    fig.suptitle("Calibration reliability on CARE Farm A (sigmoid and Laplace coincide)",
                  y=1.02)
    fig.savefig(FIG_DIR / "fig2_reliability.pdf")
    fig.savefig(FIG_DIR / "fig2_reliability.png")
    plt.close(fig)
    print("  [OK] fig2_reliability.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: Event-40 timeline (generator bearing, inverse-ranking recovery)
# ─────────────────────────────────────────────────────────────────────────

def fig3_event40_timeline():
    with (RESULTS_DIR / "care_ensemble_v7_per_event_scores.json").open() as f:
        v7 = json.load(f)
    rec = v7["40"]
    v7_scores = np.asarray(rec["scores"])
    y_event = np.asarray(rec["y_event"])
    y_status = np.asarray(rec["y_status"])

    # PINN-v2 posteriors for event 40 are not stored per-row in current
    # artifacts; we plot the v7 score and the event labels and note the AUC
    # values (v7: 0.300 on event-window; PINN-v2 partially recovers to 0.52) in the
    # caption of the paper.
    n = len(v7_scores)
    t = np.arange(n) / 6   # hours (10-min rows)

    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    # normalise v7 score to [0,1] via percentile for visual clarity
    v7_norm = pd.Series(v7_scores).rank(pct=True).to_numpy()
    ax.plot(t, v7_norm, color="#1f77b4", lw=1.0, alpha=0.8,
              label="v7 base score (rank-normalised)")
    # overlay labels
    ymin, ymax = -0.05, 1.15
    ax.fill_between(t, ymin, ymax, where=(y_event == 1),
                      color="#d62728", alpha=0.18, label="event-window label = 1")
    ax.fill_between(t, ymin, ymax, where=(y_status == 1),
                      color="#ff7f0e", alpha=0.10, label="status_type_id $\\neq 0$")

    ax.set_xlim(0, t.max())
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("hours since start of event-40 prediction window")
    ax.set_ylabel("v7 score (rank percentile)")
    ax.set_title("Event 40 (generator-bearing): v7 per-event AUC 0.300 "
                  "(inverts truth); PINN-v2 partially recovers to 0.52")
    ax.legend(loc="upper right", frameon=True, fontsize=7.5)
    ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(FIG_DIR / "fig3_event40_timeline.pdf")
    fig.savefig(FIG_DIR / "fig3_event40_timeline.png")
    plt.close(fig)
    print("  [OK] fig3_event40_timeline.pdf")


def main():
    print("Generating figures…")
    fig1_per_event_scatter()
    fig2_reliability()
    fig3_event40_timeline()
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
