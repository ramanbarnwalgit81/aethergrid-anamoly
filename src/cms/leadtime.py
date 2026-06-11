"""
Early-warning lead-time and per-fault-type detectability.

For each anomaly dataset we score the prediction frame, smooth it, and find the
first timestamp where the criticality counter raises a sustained alarm. Lead time
= hours between that first alarm and the operator's logged event_start. Positive
lead = we flagged the developing fault BEFORE the operator window — the
operationally meaningful quantity.

We also group detection rate and lead time by fault subsystem (parsed from the
CARE event_description), answering "which fault types are detectable, and how
early?". Real runs only; writes JSON + a figure to docs/results/cms/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.cms import data as D
from src.cms.care_metric import CRIT_THRESHOLD
from src.cms.models import make_scorer
from src.cms.preprocess import Preprocessor

RESULTS_DIR = Path("docs/results/cms")
STEP_HOURS = 10.0 / 60.0  # 10-min resolution


def _subsystem(desc: str) -> str:
    d = (desc or "").lower()
    for key, name in [("gear", "gearbox"), ("bearing", "bearing"),
                      ("transformer", "transformer"), ("generator", "generator"),
                      ("converter", "converter"), ("pitch", "pitch"),
                      ("blade", "blade"), ("hydraul", "hydraulic"),
                      ("cool", "cooling"), ("yaw", "yaw")]:
        if key in d:
            return name
    return "other"


def _first_alarm_idx(alerts, thr=CRIT_THRESHOLD):
    crit = 0
    for i, a in enumerate(alerts):
        crit = crit + 1 if a else max(0, crit - 1)
        if crit >= thr:
            return i
    return None


def _ewma(s, alpha=0.05):
    s = np.asarray(s, float); out = np.empty_like(s)
    if len(s) == 0:
        return s
    out[0] = s[0]
    for i in range(1, len(s)):
        out[i] = alpha * s[i] + (1 - alpha) * out[i - 1]
    return out


def run(farms, model_name="Ensemble", quantile=0.99):
    per_fault = defaultdict(lambda: {"n": 0, "detected": 0, "lead_hours": []})
    rows = []
    for farm in farms:
        for ds in D.iter_care(farm):
            if ds.label != "anomaly":
                continue
            tr = ds.train_frame(producing_only=True)
            pr = ds.predict_frame(producing_only=False)
            if len(tr) < 200 or len(pr) < 50:
                continue
            pre = Preprocessor(ds.feature_cols).fit(tr)
            if pre.n_features == 0:
                continue
            sc = make_scorer(model_name)
            sc.fit(pre.transform(tr))
            s_tr = _ewma(sc.score(pre.transform(tr)))
            s_pr = _ewma(sc.score(pre.transform(pr)))
            thr = float(np.quantile(s_tr, quantile))
            alerts = s_pr >= thr
            pidx = pr.index.to_numpy()
            ids = ds.df.loc[pidx, "id"].to_numpy()

            fa = _first_alarm_idx(alerts)
            sub = _subsystem(ds.description)
            per_fault[sub]["n"] += 1
            lead = None
            if fa is not None:
                per_fault[sub]["detected"] += 1
                lead = float((ds.event_start_id - ids[fa]) * STEP_HOURS)
                per_fault[sub]["lead_hours"].append(lead)
            rows.append({"farm": farm, "event_id": ds.event_id, "subsystem": sub,
                         "detected": fa is not None,
                         "lead_hours": round(lead, 1) if lead is not None else None})

    summary = {}
    for sub, v in per_fault.items():
        leads = v["lead_hours"]
        summary[sub] = {
            "n_events": v["n"], "detected": v["detected"],
            "detection_rate": round(v["detected"] / v["n"], 3) if v["n"] else 0,
            "median_lead_hours": round(float(np.median(leads)), 1) if leads else None,
            "mean_lead_hours": round(float(np.mean(leads)), 1) if leads else None,
        }
    all_leads = [r["lead_hours"] for r in rows if r["lead_hours"] is not None]
    result = {
        "model": model_name, "farms": farms, "quantile": quantile,
        "n_anomaly_events": len(rows),
        "n_detected": int(sum(r["detected"] for r in rows)),
        "median_lead_hours_all": round(float(np.median(all_leads)), 1) if all_leads else None,
        "positive_lead_fraction": round(float(np.mean([l > 0 for l in all_leads])), 3) if all_leads else None,
        "per_subsystem": summary,
        "per_event": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"leadtime_{''.join(farms)}_{model_name}.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\nLead-time / per-fault ({model_name}) farms {','.join(farms)}")
    print(f"  detected {result['n_detected']}/{result['n_anomaly_events']} events; "
          f"median lead {result['median_lead_hours_all']} h; "
          f"positive-lead fraction {result['positive_lead_fraction']}")
    for sub, v in sorted(summary.items(), key=lambda kv: -kv[1]["n_events"]):
        print(f"  {sub:12s} n={v['n_events']:<3} det_rate={v['detection_rate']} "
              f"median_lead={v['median_lead_hours']}h")
    print(f"[OK] wrote {out}")

    _plot(summary, result, RESULTS_DIR / f"leadtime_{''.join(farms)}_{model_name}.png")
    return result


def _plot(summary, result, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    subs = list(summary.keys())
    rates = [summary[s]["detection_rate"] for s in subs]
    leads = [summary[s]["median_lead_hours"] or 0 for s in subs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].barh(subs, rates, color="#2c7fb8")
    ax[0].set_xlabel("Detection rate"); ax[0].set_xlim(0, 1)
    ax[0].set_title("Per-fault-type detection rate")
    ax[1].barh(subs, leads, color="#d95f0e")
    ax[1].set_xlabel("Median lead time (h)")
    ax[1].set_title("Early-warning lead time before logged failure")
    fig.suptitle(f"{result['model']} — {result['n_detected']}/{result['n_anomaly_events']} "
                 f"events detected, median lead {result['median_lead_hours_all']} h")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"[OK] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="A,B")
    ap.add_argument("--model", default="Ensemble")
    ap.add_argument("--quantile", type=float, default=0.99)
    args = ap.parse_args()
    run([f.strip() for f in args.farms.split(",")], args.model, args.quantile)


if __name__ == "__main__":
    main()
