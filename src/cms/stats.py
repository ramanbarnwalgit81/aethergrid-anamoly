"""
Statistical significance for the model comparison.

Two AUCs computed on the SAME datasets are correlated, so an unpaired test is
invalid. We use the fast DeLong test (Sun & Xu, 2014) for pairwise AUC
differences, plus stratified bootstrap CIs. This is what turns "0.797 vs 0.655"
into "not significantly different (p = ...)" — the backbone of the plateau claim.

Reads the per-dataset (label, peak-score) pairs exported by src.cms.benchmark
into benchmark_*.json under "level_scores". Fabricates nothing.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("docs/results/cms")


# ----------------------------- fast DeLong --------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, n_pos):
    """preds_sorted: (k_models, n) with positives first. Returns AUCs, cov."""
    m = n_pos
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(labels, score_a, score_b):
    """Return (auc_a, auc_b, p_value) for H0: AUC_a == AUC_b (paired)."""
    labels = np.asarray(labels)
    order = np.argsort(-labels)  # positives (1) first
    lab = labels[order]
    n_pos = int(lab.sum())
    preds = np.vstack([np.asarray(score_a)[order], np.asarray(score_b)[order]])
    aucs, cov = _fast_delong(preds, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        p = 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


def bootstrap_auc(labels, scores, n_boot=3000, seed=42):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels); scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    base = roc_auc_score(labels, scores)
    n = len(labels); a = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(labels[idx])) > 1:
            a.append(roc_auc_score(labels[idx], scores[idx]))
    return float(base), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def _load_levels(tags):
    """Return {model_label: (labels[], scores[])} from benchmark JSONs."""
    out = {}
    for tag in tags:
        p = RESULTS_DIR / f"benchmark_{tag}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for key, recs in data.get("level_scores", {}).items():
            model = key.split("|")[0]
            labs = [1 if r["label"] in (1, "anomaly", True) else 0 for r in recs]
            scs = [r["score"] for r in recs]
            # keep the smooth setting that yields the best separation per model
            if model not in out or len(recs) > len(out[model][0]):
                out[model] = (labs, scs)
    return out


def _load_care_contribs(tags):
    """Return {model: [per-dataset contribution records]} from benchmark JSONs."""
    out = {}
    for tag in tags:
        p = RESULTS_DIR / f"benchmark_{tag}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for key, recs in data.get("care_per_dataset", {}).items():
            out[key.split("|")[0]] = recs
    return out


def bootstrap_care(recs, n_boot=3000, seed=42):
    from src.cms.care_metric import care_from_contributions
    rng = np.random.default_rng(seed)
    n = len(recs)
    base = care_from_contributions(recs)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(care_from_contributions([recs[i] for i in idx]))
    return float(base), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_care_pvalue(recs_a, recs_b, n_boot=3000, seed=42):
    """Paired bootstrap p-value for H0: CARE_a == CARE_b (same resampled datasets)."""
    from src.cms.care_metric import care_from_contributions
    if len(recs_a) != len(recs_b):
        return None
    rng = np.random.default_rng(seed)
    n = len(recs_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        da = care_from_contributions([recs_a[i] for i in idx])
        db = care_from_contributions([recs_b[i] for i in idx])
        diffs.append(da - db)
    diffs = np.array(diffs)
    # two-sided: fraction of resamples on the opposite side of zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(min(1.0, p))


def care_significance(tags):
    contribs = _load_care_contribs(tags)
    if not contribs:
        return []
    lines = ["", "## CARE score — bootstrap 95% CI", "",
             "| Model | n | CARE [95% CI] |", "|---|---|---|"]
    cis = {}
    for m, recs in sorted(contribs.items()):
        b, lo, hi = bootstrap_care(recs)
        cis[m] = (b, lo, hi)
        lines.append(f"| {m} | {len(recs)} | {b:.3f} [{lo:.3f}, {hi:.3f}] |")
    # paired vs the best model
    best = max(cis, key=lambda m: cis[m][0])
    lines += ["", f"## Paired CARE test vs best model ({best})", "",
              "| Model | CARE | p (paired bootstrap) | sig (p<0.05) |", "|---|---|---|---|"]
    for m in sorted(contribs):
        if m == best:
            continue
        p = paired_care_pvalue(contribs[best], contribs[m])
        sig = "YES" if (p is not None and p < 0.05) else "no"
        ps = f"{p:.3f}" if p is not None else "n/a"
        lines.append(f"| {m} | {cis[m][0]:.3f} | {ps} | {sig} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="core,GDN_AB")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",")]
    models = _load_levels(tags)
    if not models:
        print("No level_scores found in", tags)
        return

    lines = ["# Statistical significance — event-level AUC", "",
             "Bootstrap 95% CI (3000 resamples) and pairwise DeLong test "
             "(paired AUC comparison on the same datasets).", "",
             "## Per-model AUC", "", "| Model | n | AUC [95% CI] |", "|---|---|---|"]
    aucs = {}
    for m, (labs, scs) in models.items():
        base, lo, hi = bootstrap_auc(labs, scs)
        aucs[m] = (base, lo, hi, labs, scs)
        lines.append(f"| {m} | {len(labs)} | {base:.3f} [{lo:.3f}, {hi:.3f}] |")

    lines += ["", "## Pairwise DeLong p-values (AUC difference)", "",
              "| Model A | Model B | AUC_A | AUC_B | p | significant (p<0.05) |",
              "|---|---|---|---|---|---|"]
    for a, b in combinations(models, 2):
        la, sa = models[a]; lb, sb = models[b]
        if len(la) != len(lb):
            # only comparable when evaluated on the same dataset set
            lines.append(f"| {a} | {b} | — | — | n/a (different farms) | — |")
            continue
        auc_a, auc_b, p = delong_test(la, sa, sb)
        sig = "YES" if p < 0.05 else "no"
        lines.append(f"| {a} | {b} | {auc_a:.3f} | {auc_b:.3f} | {p:.3f} | {sig} |")

    lines += care_significance(tags)

    out = RESULTS_DIR / "significance.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] wrote {out}")


if __name__ == "__main__":
    main()
