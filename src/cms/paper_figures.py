"""
Generate the three figures for the corrected-protocol section (paper Sec. 5.8):
  fig_cms_detectors  : detector CARE + event-AUC with 95% CIs and the 0.66 line
  fig_cms_labelgap   : status vs event vs precursor per-row AUC distributions
  fig_cms_gdn_graph  : the GDN learned inter-sensor dependency graph (Farm A)

All inputs are real run artifacts in docs/results/cms/. Outputs PDF+PNG (300 dpi)
to docs/paper/figures/. Nothing is fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cms.stats import bootstrap_auc
from src.cms.care_metric import care_from_contributions

RES = Path("docs/results/cms")
FIG = Path("docs/paper/figures")
PUBLISHED_AE = 0.66


def _bootstrap_care_ci(recs, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    base = care_from_contributions(recs)
    vals = [care_from_contributions([recs[i] for i in rng.integers(0, len(recs), len(recs))])
            for _ in range(n_boot)]
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fig_detectors():
    d = json.loads((RES / "benchmark_baselines.json").read_text())
    rows, level, cpd = d["rows"], d["level_scores"], d["care_per_dataset"]
    best = {}
    for r in rows:
        m = r["model"]
        if m not in best or r["agg_CARE"] > best[m]["agg_CARE"]:
            best[m] = r
    models, care, care_lo, care_hi, auc, auc_lo, auc_hi = [], [], [], [], [], [], []
    for m, r in best.items():
        ck = next((k for k in cpd if k.split("|")[0] == m), None)
        lk = f"{m}|{r['smooth']}"
        if ck is None or lk not in level:
            continue
        b, lo, hi = _bootstrap_care_ci(cpd[ck])
        labs = [1 if x["label"] in (1, "anomaly", True) else 0 for x in level[lk]]
        scs = [x["score"] for x in level[lk]]
        ab, alo, ahi = bootstrap_auc(labs, scs)
        models.append(m); care.append(b); care_lo.append(lo); care_hi.append(hi)
        auc.append(ab); auc_lo.append(alo); auc_hi.append(ahi)
    order = np.argsort(care)
    models = [models[i] for i in order]
    def srt(x): return [x[i] for i in order]
    care, care_lo, care_hi = srt(care), srt(care_lo), srt(care_hi)
    auc, auc_lo, auc_hi = srt(auc), srt(auc_lo), srt(auc_hi)
    y = np.arange(len(models))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    ax[0].errorbar(care, y, xerr=[np.array(care) - care_lo, np.array(care_hi) - care],
                   fmt="o", color="#1f4e79", capsize=3)
    ax[0].axvline(PUBLISHED_AE, ls="--", color="#c00", lw=1.2)
    ax[0].text(PUBLISHED_AE, len(models) - 0.4, " published AE 0.66", color="#c00",
               fontsize=7, va="top")
    ax[0].set_yticks(y); ax[0].set_yticklabels(models, fontsize=8)
    ax[0].set_xlabel("Official CARE score"); ax[0].set_title("(a) CARE (95% CI)", fontsize=9)
    ax[1].errorbar(auc, y, xerr=[np.array(auc) - auc_lo, np.array(auc_hi) - auc],
                   fmt="s", color="#2c7a2c", capsize=3)
    ax[1].axvline(0.5, ls=":", color="gray", lw=1)
    ax[1].set_xlabel("Event-level AUC"); ax[1].set_title("(b) AUC (95% CI)", fontsize=9)
    for a in ax:
        a.grid(axis="x", alpha=0.3)
    fig.suptitle("Detector comparison on CARE (95 datasets): a performance plateau",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig_cms_detectors")


def fig_labelgap():
    d = json.loads((RES / "label_sensitivity_ABC.json").read_text())
    pd_ = d["per_dataset"]
    cats = [("status", "Operator\nstatus"), ("event", "Event\nwindow"),
            ("precursor", "Precursor\n(predictive)")]
    data = [np.array(pd_[c]) for c, _ in cats]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black"))
    for patch, col in zip(bp["boxes"], ["#2c7a2c", "#b0843b", "#9b2226"]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xticklabels([lbl for _, lbl in cats], fontsize=8)
    ax.set_ylabel("Per-row detection AUC")
    med = [float(np.median(x)) for x in data]
    ax.annotate("", xy=(1, med[0]), xytext=(3, med[2]),
                arrowprops=dict(arrowstyle="<->", color="#9b2226", lw=1.3))
    ax.text(2.0, (med[0] + med[2]) / 2, f"median gap\n{med[0]-med[2]:.2f}",
            color="#9b2226", fontsize=8, ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="#9b2226", alpha=0.9))
    ax.set_title("Same scores, three labels (Mahalanobis, 42 events)", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_cms_labelgap")


# Farm-A semantic sensor names (from CARE feature_description; legacy adapter map).
FARM_A_MAP = {
    "wind_speed_3_avg": "wind_speed", "sensor_0_avg": "ambient_T",
    "power_30_avg": "power", "sensor_18_avg": "gen_rpm",
    "sensor_12_avg": "gearbox_oil_T", "sensor_11_avg": "gearbox_brg_T",
    "sensor_13_avg": "gen_brg_DE_T", "sensor_14_avg": "gen_brg_NDE_T",
    "sensor_15_avg": "gen_wind_T", "sensor_7_avg": "nacelle_T",
    "sensor_1_avg": "wind_dir", "sensor_5_avg": "pitch",
    "reactive_power_27_avg": "react_power",
}


def fig_gdn_graph():
    """Fit a small GDN on Farm A's named drivetrain/electrical channels and draw
    the learned top-k inter-sensor dependency graph."""
    import torch
    from src.cms import data as D
    from src.cms.preprocess import Preprocessor
    from src.cms.models_graph import GDNScorer

    ds = next(d for d in D.iter_care("A") if d.label == "anomaly")
    cols = [c for c in FARM_A_MAP if c in ds.feature_cols]
    names = [FARM_A_MAP[c] for c in cols]
    tr = ds.train_frame(producing_only=True)
    pre = Preprocessor(cols).fit(tr)
    names = [FARM_A_MAP[c] for c in pre.kept_cols]
    X = pre.transform(tr)
    g = GDNScorer(epochs=15, reduce_above=999, topk=3, max_windows=6000)
    g.fit(X)
    with torch.no_grad():
        emb = torch.nn.functional.normalize(g.model.emb.weight, dim=1)
        sim = (emb @ emb.t())
        sim.fill_diagonal_(-1e9)
        nbr = torch.topk(sim, min(3, sim.shape[0] - 1), dim=1).indices.numpy()
    n = len(names)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.c_[np.cos(ang), np.sin(ang)]
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for i in range(n):
        for j in nbr[i]:
            ax.annotate("", xy=pos[j], xytext=pos[i],
                        arrowprops=dict(arrowstyle="-|>", color="#888", alpha=0.6, lw=0.9))
    ax.scatter(pos[:, 0], pos[:, 1], s=420, c="#1f4e79", zorder=3, edgecolors="white")
    for i, nm in enumerate(names):
        ax.text(pos[i, 0] * 1.18, pos[i, 1] * 1.18, nm, ha="center", va="center",
                fontsize=7.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    ax.set_title("GDN learned inter-sensor dependency graph (CARE Farm A)\n"
                 "directed edge = top-3 learned neighbours", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_cms_gdn_graph")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {FIG / name}.pdf/.png")


def main():
    fig_detectors()
    fig_labelgap()
    try:
        fig_gdn_graph()
    except Exception as e:
        print(f"[WARN] GDN graph figure skipped: {e}")


if __name__ == "__main__":
    main()
