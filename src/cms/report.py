"""
Build the final comparison report from a benchmark JSON produced by
src.cms.benchmark — including bootstrap 95% CIs on event-level AUC (resampling
datasets), the best CARE per model, and the comparison to published SOTA.

Pure post-processing: reads only real run artifacts, fabricates nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("docs/results/cms")
PUBLISHED_CARE = {"Random": 0.50, "Isolation Forest (paper)": 0.45,
                  "Autoencoder (paper, SOTA)": 0.66}


def bootstrap_auc(labels, scores, n_boot=2000, seed=42):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(labels)
    base = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = (np.percentile(aucs, [2.5, 97.5]) if aucs else (np.nan, np.nan))
    return float(base), float(lo), float(hi)


def build(tags):
    if isinstance(tags, str):
        tags = [tags]
    rows, level = [], {}
    farms_seen = []
    for tag in tags:
        path = RESULTS_DIR / f"benchmark_{tag}.json"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        data = json.loads(path.read_text())
        rows.extend(data["rows"])
        level.update(data.get("level_scores", {}))
        for f in data["config"]["farms"]:
            if f not in farms_seen:
                farms_seen.append(f)
    data = {"config": {"farms": farms_seen}}

    # Best CARE per model (across smooth/quantile) and its AUC.
    best_care = {}
    for r in rows:
        m = r["model"]
        if m not in best_care or r["agg_CARE"] > best_care[m]["agg_CARE"]:
            best_care[m] = r

    # Bootstrap AUC per model (use the smooth setting of its best-CARE row).
    auc_ci = {}
    for m, r in best_care.items():
        key = f"{m}|{r['smooth']}"
        if key in level and level[key]:
            labs = [1 if d["label"] in (1, "anomaly", True) else 0 for d in level[key]]
            scs = [d["score"] for d in level[key]]
            auc_ci[m] = bootstrap_auc(labs, scs)
        else:
            auc_ci[m] = (r["agg_AUC"], float("nan"), float("nan"))

    lines = [
        "# CARE benchmark — final results (clean cms pipeline)", "",
        f"Merged tags: {', '.join(tags)} | farms: {','.join(farms_seen)}",
        "",
        "Published reference (official aggregate CARE score): "
        + ", ".join(f"**{k} = {v}**" for k, v in PUBLISHED_CARE.items()),
        "",
        "## Best operating point per model (aggregate over all farms)", "",
        "| Model | Best CARE | event-AUC [95% CI] | smooth | q | tp/fp/fn | per-farm CARE |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, r in sorted(best_care.items(), key=lambda kv: -kv[1]["agg_CARE"]):
        base, lo, hi = auc_ci[m]
        ci = f"{base:.3f} [{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else f"{base:.3f}"
        pf = " ".join(f"{k}:{v}" for k, v in r["per_farm_CARE"].items())
        beat = " ✅" if r["agg_CARE"] > 0.66 else ""
        lines.append(f"| {m} | **{r['agg_CARE']:.4f}**{beat} | {ci} | {r['smooth']} "
                     f"| {r['quantile']} | {r['ev_tp']}/{r['ev_fp']}/{r['ev_fn']} | {pf} |")

    # Calibrated operating point (fp<=1) recomputed from merged rows per model.
    lines += ["", "## Calibrated operating point (most sensitive threshold with event-FP<=1)",
              "", "| Model | CARE | event-AUC | smooth | q | tp/fp/fn |",
              "|---|---|---|---|---|---|"]
    by_m = {}
    for r in rows:
        by_m.setdefault(r["model"], []).append(r)
    for m, grp in by_m.items():
        grp_sorted = sorted(grp, key=lambda r: r["quantile"])
        pick = next((r for r in grp_sorted if r["ev_fp"] <= 1), None) or \
            min(grp_sorted, key=lambda r: r["ev_fp"])
        beat = " ✅" if pick["agg_CARE"] > 0.66 else ""
        lines.append(f"| {m} | {pick['agg_CARE']:.4f}{beat} | {pick['agg_AUC']:.3f} "
                     f"| {pick['smooth']} | {pick['quantile']} | "
                     f"{pick['ev_tp']}/{pick['ev_fp']}/{pick['ev_fn']} |")

    out = RESULTS_DIR / "REPORT_final.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode("ascii"))
    print(f"\n[OK] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="core,GDN_AB,ABC",
                    help="comma list of benchmark tags to merge")
    args = ap.parse_args()
    build([t.strip() for t in args.tags.split(",") if t.strip()])


if __name__ == "__main__":
    main()
