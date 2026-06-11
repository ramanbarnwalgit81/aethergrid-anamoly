"""
Consolidated CARE benchmark run.

For each model: evaluate every requested farm, then pool all per-dataset scores
across farms into ONE aggregate CARE computation — matching how the original
CARE paper reports a single benchmark-wide score (Autoencoder = 0.66, Isolation
Forest ~0.45, Random = 0.5). Also reports per-farm CARE and event-level AUC.

Outputs (real, reproducible — nothing fabricated):
  docs/results/cms/benchmark_<tag>.json   full numbers
  docs/results/cms/benchmark_<tag>.md     comparison table
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.cms.care_metric import compute_care
from src.cms.evaluate import RESULTS_DIR, evaluate_farm

# Published reference points (original CARE paper, aggregate benchmark CARE score).
PUBLISHED = {"Random": 0.50, "IsolationForest(paper)": 0.45, "Autoencoder(paper)": 0.66}


def run(models, farms, smooth, quantile=0.995):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    per_model_detail = {}

    for model_name in models:
        all_ds = []
        level_scores, level_labels = [], []
        per_farm = {}
        for farm in farms:
            print(f"\n=== {model_name} | Farm {farm} | smooth={smooth} q={quantile} ===", flush=True)
            res = evaluate_farm(farm, model_name, smooth_alpha=smooth,
                                quantile=quantile, verbose=False)
            all_ds.extend(res["_ds_scores"])
            level_scores.extend(res["_level_scores"])
            level_labels.extend(res["_level_labels"])
            per_farm[farm] = {
                "care": res["care"]["care_score"],
                "event_auc": res["event_level_auc"],
                "event_ap": res["event_level_ap"],
                "n_datasets": res["n_datasets"],
            }
            print(f"    Farm {farm}: CARE={res['care']['care_score']} "
                  f"AUC={res['event_level_auc']} (n={res['n_datasets']})", flush=True)

        agg_care = compute_care(all_ds)
        y = np.array(level_labels)
        s = np.array(level_scores)
        agg_auc = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
        agg_ap = float(average_precision_score(y, s)) if len(set(y)) > 1 else float("nan")

        per_model_detail[model_name] = {
            "aggregate_care": agg_care,
            "aggregate_event_auc": round(agg_auc, 4),
            "aggregate_event_ap": round(agg_ap, 4),
            "per_farm": per_farm,
        }
        rows.append({
            "model": model_name,
            "agg_CARE": agg_care["care_score"],
            "agg_AUC": round(agg_auc, 4),
            "agg_AP": round(agg_ap, 4),
            **{f"CARE_{f}": per_farm[f]["care"] for f in farms},
            **{f"AUC_{f}": per_farm[f]["event_auc"] for f in farms},
        })
        print(f"  >> {model_name}: AGGREGATE CARE={agg_care['care_score']}  "
              f"AUC={round(agg_auc,4)}", flush=True)

    tag = f"farms{''.join(farms)}_s{smooth}"
    out_json = RESULTS_DIR / f"benchmark_{tag}.json"
    out_json.write_text(json.dumps({
        "config": {"models": models, "farms": farms, "smooth_alpha": smooth},
        "published_reference": PUBLISHED,
        "results": per_model_detail,
        "table": rows,
    }, indent=2))

    # Markdown table
    md = ["# CARE benchmark — clean cms pipeline", "",
          f"Farms: {', '.join(farms)} | EWMA smooth alpha: {smooth} | "
          f"train-quantile threshold (leak-free)", "",
          "Published reference (aggregate CARE): "
          + ", ".join(f"{k}={v}" for k, v in PUBLISHED.items()), "",
          "| Model | Aggregate CARE | Aggregate event-AUC | "
          + " | ".join(f"CARE {f}" for f in farms) + " |",
          "|---|---|---|" + "---|" * len(farms)]
    for r in sorted(rows, key=lambda x: -x["agg_CARE"]):
        md.append(f"| {r['model']} | **{r['agg_CARE']}** | {r['agg_AUC']} | "
                  + " | ".join(str(r[f'CARE_{f}']) for f in farms) + " |")
    out_md = RESULTS_DIR / f"benchmark_{tag}.md"
    out_md.write_text("\n".join(md))

    print("\n" + "=" * 64)
    print("AGGREGATE BENCHMARK (vs published Autoencoder CARE=0.66)")
    print("=" * 64)
    for r in sorted(rows, key=lambda x: -x["agg_CARE"]):
        flag = "  <-- beats AE 0.66" if r["agg_CARE"] > 0.66 else ""
        print(f"  {r['model']:<16} CARE={r['agg_CARE']:.4f}  AUC={r['agg_AUC']:.4f}{flag}")
    print(f"\n[OK] wrote {out_json}\n[OK] wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="PCARecon,Mahalanobis,Ensemble")
    ap.add_argument("--farms", default="A,B,C")
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--quantile", type=float, default=0.995)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    farms = [f.strip() for f in args.farms.split(",") if f.strip()]
    run(models, farms, args.smooth, args.quantile)


if __name__ == "__main__":
    main()
