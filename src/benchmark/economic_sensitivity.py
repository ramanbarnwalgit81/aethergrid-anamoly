"""Economic sensitivity sweep: CF x PPA x collateral-damage multiplier.

The paper's Table of per-fault savings fixes capacity factor (CF) = 35%,
PPA = $45/MWh, and a 1.7x reactive-to-predictive collateral multiplier
(Carroll et al. 2016). A reviewer asked whether the "alpha = 0.10 optimal"
and the headline savings numbers survive variation across plausible ranges.
We sweep CF in {0.25, 0.30, 0.35, 0.40, 0.45}, PPA in {30, 45, 60, 80} USD/MWh,
and the multiplier in {1.3, 1.5, 1.7, 2.0, 2.5} and report the 5th and 95th
percentile of the headline "predictive savings per caught gearbox event".

Usage:  python -m src.benchmark.economic_sensitivity
Outputs: docs/results/economic_sensitivity.json
"""
from __future__ import annotations
from pathlib import Path
import json
import itertools
import numpy as np

RESULTS_DIR = Path("docs/results")

# Paper's base-case assumptions
RATED_KW = 2000.0
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365

# Per-fault reactive parameters (from Carroll et al. 2016 and Wind Systems 2023)
# The "reactive_repair_cost" includes labour, parts, crane, collateral damage,
# assuming a 1.7x multiplier over the predictive-maintenance base cost.
BASE_PREDICTIVE_COSTS = {
    "Gearbox failure":           {"base_repair": 350_000, "down_days": 14},
    "Transformer failure":       {"base_repair": 250_000, "down_days": 10},
    "Generator bearing failure": {"base_repair": 100_000, "down_days": 3},
    "Hydraulic group":           {"base_repair":  30_000, "down_days": 0.75},
}


def run_scenario(cf: float, ppa: float, mult: float) -> dict:
    downtime_per_day = RATED_KW * cf * HOURS_PER_DAY * ppa / 1000.0
    out = {}
    for fault, p in BASE_PREDICTIVE_COSTS.items():
        pred_down = p["down_days"] * downtime_per_day
        pred_total = p["base_repair"] + pred_down
        react_repair = p["base_repair"] * mult
        react_down = p["down_days"] * downtime_per_day * mult
        react_total = react_repair + react_down
        out[fault] = {
            "predictive_total": round(pred_total, 0),
            "reactive_total": round(react_total, 0),
            "savings": round(react_total - pred_total, 0),
        }
    return out


def main():
    cf_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
    ppa_grid = [30, 45, 60, 80]
    mult_grid = [1.3, 1.5, 1.7, 2.0, 2.5]

    # Collect savings distribution per fault type across the full grid
    per_fault_savings = {f: [] for f in BASE_PREDICTIVE_COSTS}
    paper_baseline = {"cf": 0.35, "ppa": 45, "mult": 1.7}

    for cf, ppa, mult in itertools.product(cf_grid, ppa_grid, mult_grid):
        r = run_scenario(cf, ppa, mult)
        for f, x in r.items():
            per_fault_savings[f].append(x["savings"])

    summary = {}
    for f, arr in per_fault_savings.items():
        a = np.asarray(arr)
        summary[f] = {
            "min":       int(a.min()),
            "p05":       int(np.percentile(a, 5)),
            "median":    int(np.percentile(a, 50)),
            "p95":       int(np.percentile(a, 95)),
            "max":       int(a.max()),
            "baseline":  int(run_scenario(**paper_baseline)[f]["savings"]),
        }
    # Robustness: fraction of scenarios where gearbox savings exceed 100k
    gearbox_ge_100k = float(np.mean(np.asarray(per_fault_savings["Gearbox failure"]) >= 100_000))
    gearbox_ge_150k = float(np.mean(np.asarray(per_fault_savings["Gearbox failure"]) >= 150_000))

    out = {
        "grid": {"capacity_factor": cf_grid, "ppa_usd_per_mwh": ppa_grid,
                    "collateral_multiplier": mult_grid},
        "baseline_scenario": paper_baseline,
        "per_fault_savings_distribution": summary,
        "gearbox_robustness": {
            "P[savings >= 100k]": round(gearbox_ge_100k, 3),
            "P[savings >= 150k]": round(gearbox_ge_150k, 3),
        },
        "n_scenarios": len(per_fault_savings["Gearbox failure"]),
    }
    (RESULTS_DIR / "economic_sensitivity.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"Evaluated {out['n_scenarios']} scenarios (CF x PPA x multiplier grid).\n")
    print(f"{'Fault':<30} {'P05':>10} {'Median':>10} {'P95':>10} {'Baseline':>10}")
    print("-" * 75)
    for f, s in summary.items():
        print(f"{f:<30} ${s['p05']:>9,} ${s['median']:>9,} ${s['p95']:>9,} ${s['baseline']:>9,}")
    print(f"\nGearbox P[savings >= 100k] = {gearbox_ge_100k:.2%}")
    print(f"Gearbox P[savings >= 150k] = {gearbox_ge_150k:.2%}")
    print("\n[OK] Saved: docs/results/economic_sensitivity.json")


if __name__ == "__main__":
    main()
