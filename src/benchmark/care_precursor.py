"""
CARE-Precursor: re-labeling CARE with operator-alert-onset timestamps.

PROBLEM
-------
CARE's `event_start_id` is the **maintenance-log** timestamp of the
fault — often 2–14 days AFTER the turbine's PLC triggered the first
non-zero `status_type_id`. A predictor that catches the fault at
`event_start_id` is not actually predictive; it's retrospective.

FIX
---
For each anomaly event, scan the `status_type_id` trace and find the
FIRST non-zero entry within a configurable horizon (default 30 days)
BEFORE `event_start_id`. Call this `precursor_start_id`.

Label CARE-Precursor with:
  anomaly window = [precursor_start_id, event_end_id]

This gives a **genuinely predictive** benchmark: a model that
detects anomaly inside this window is catching the problem at or
BEFORE the operator's own alert — which is exactly what predictive
maintenance means.

OUTPUT
------
A new `event_info_precursor.csv` per farm with added columns:
  precursor_start_id, precursor_start, precursor_lead_time_hours

And a summary JSON per farm with statistics:
  lead-time distribution, fault-type-specific lead times,
  coverage expansion (how much larger the anomaly window is).

USAGE
-----
    python -m src.benchmark.care_precursor --farm A --horizon-days 30
    python -m src.benchmark.care_precursor --farm B
    python -m src.benchmark.care_precursor --farm C
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


CARE_BASE = Path("data/real_scada/care/extracted")
OUT_DIR = Path("data/benchmark/care_precursor")


def find_precursor(df: pd.DataFrame, event_start_id: int,
                    horizon_rows: int,
                    min_sustained_rows: int = 12) -> tuple:
    """
    Find the first row within [event_start_id - horizon_rows, event_start_id)
    where status_type_id != 0 AND the non-zero state is SUSTAINED for at
    least `min_sustained_rows` subsequent rows.

    Default: 12 rows = 2 hours at 10-min resolution. This filters out brief
    curtailment/communication noise and captures actual persistent
    degradation precursors.

    Strategy: scan BACKWARDS from event_start_id. Find the latest point
    before event_start_id where the turbine was still in a sustained
    NORMAL state (status==0 for >= min_sustained_rows consecutively).
    The precursor starts one row after that "last normal" point.

    Returns (precursor_start_id, found_in_horizon).
    """
    start = max(0, event_start_id - horizon_rows)
    window = df.iloc[start:event_start_id]
    if "status_type_id" not in df.columns or len(window) < min_sustained_rows:
        return event_start_id, False

    status = window["status_type_id"].fillna(0).astype(int).values
    is_normal = (status == 0).astype(int)

    # Find runs of consecutive normal rows. We want the LAST run of length
    # >= min_sustained_rows that ends BEFORE event_start_id — the precursor
    # starts at the end of that run + 1.
    n = len(is_normal)
    last_sustained_normal_end = -1  # relative index
    run_len = 0
    for i in range(n):
        if is_normal[i] == 1:
            run_len += 1
        else:
            if run_len >= min_sustained_rows:
                last_sustained_normal_end = i - 1  # end of the run
            run_len = 0
    # Handle trailing normal run (turbine normal up to event_start)
    if run_len >= min_sustained_rows:
        last_sustained_normal_end = n - 1

    if last_sustained_normal_end == -1:
        # No sustained normal period found within horizon — use start of horizon
        return start, True
    if last_sustained_normal_end >= n - 1:
        # Turbine was normal right up to event_start_id — no precursor
        return event_start_id, False

    precursor_idx = start + last_sustained_normal_end + 1
    return precursor_idx, True


def process_farm(farm: str, horizon_days: float = 30.0,
                  res_minutes: int = 10) -> dict:
    """
    Build CARE-Precursor labels for one wind farm.

    Args:
        farm: "A", "B", or "C".
        horizon_days: max look-back period in days.
        res_minutes: SCADA resolution.
    """
    farm_dir = CARE_BASE / f"Wind Farm {farm}" / f"Wind Farm {farm}"
    datasets_dir = farm_dir / "datasets"
    event_info_path = farm_dir / "event_info.csv"

    if not event_info_path.exists():
        print(f"[ERR] Farm {farm} not found at {farm_dir}")
        return {}

    events = pd.read_csv(event_info_path, sep=";")
    horizon_rows = int(horizon_days * 24 * 60 / res_minutes)
    print(f"[FARM {farm}] {len(events)} events; horizon = "
          f"{horizon_days} days = {horizon_rows} rows")

    rows = []
    for _, row in events.iterrows():
        event_id = int(row["event_id"])
        csv_path = datasets_dir / f"{event_id}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, sep=";", low_memory=False,
                          usecols=lambda c: c in ["status_type_id", "time_stamp"])

        out_row = {
            "event_id": event_id,
            "event_label": row["event_label"],
            "event_description": row.get("event_description", ""),
            "event_start_id": row.get("event_start_id"),
            "event_end_id": row.get("event_end_id"),
            "event_start": row.get("event_start"),
            "event_end": row.get("event_end"),
            "precursor_start_id": row.get("event_start_id"),
            "precursor_start": row.get("event_start"),
            "precursor_lead_time_hours": 0.0,
            "precursor_found": False,
        }

        if row["event_label"] == "anomaly":
            event_start_id = int(row["event_start_id"])
            pre_id, found = find_precursor(df, event_start_id, horizon_rows)
            out_row["precursor_start_id"] = pre_id
            out_row["precursor_found"] = bool(found)
            if found and "time_stamp" in df.columns:
                out_row["precursor_start"] = df["time_stamp"].iloc[pre_id]
                lead_rows = event_start_id - pre_id
                out_row["precursor_lead_time_hours"] = \
                    lead_rows * res_minutes / 60.0

        rows.append(out_row)

    out_df = pd.DataFrame(rows)

    # Stats
    anomaly_df = out_df[out_df["event_label"] == "anomaly"]
    found_df = anomaly_df[anomaly_df["precursor_found"]]

    stats = {
        "farm": farm,
        "horizon_days": horizon_days,
        "n_events": len(out_df),
        "n_anomaly": len(anomaly_df),
        "n_precursor_found": int(len(found_df)),
        "fraction_precursor_found": float(len(found_df) / max(len(anomaly_df), 1)),
        "lead_time_hours_mean": float(found_df["precursor_lead_time_hours"].mean())
                                  if len(found_df) else 0.0,
        "lead_time_hours_median": float(found_df["precursor_lead_time_hours"].median())
                                    if len(found_df) else 0.0,
        "lead_time_hours_min": float(found_df["precursor_lead_time_hours"].min())
                                  if len(found_df) else 0.0,
        "lead_time_hours_max": float(found_df["precursor_lead_time_hours"].max())
                                  if len(found_df) else 0.0,
    }

    # Per-fault-type breakdown
    fault_breakdown = {}
    for fault, group in found_df.groupby("event_description"):
        fault_breakdown[str(fault)] = {
            "n": int(len(group)),
            "lead_time_hours_mean": float(group["precursor_lead_time_hours"].mean()),
            "lead_time_hours_median": float(group["precursor_lead_time_hours"].median()),
        }
    stats["per_fault_type"] = fault_breakdown

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"event_info_precursor_farm_{farm}.csv"
    json_path = OUT_DIR / f"precursor_stats_farm_{farm}.json"
    out_df.to_csv(csv_path, index=False, sep=";")
    with json_path.open("w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n[FARM {farm}] Stats:")
    print(f"  Precursor found: {stats['n_precursor_found']}/{stats['n_anomaly']} "
          f"({stats['fraction_precursor_found']:.1%})")
    if stats["n_precursor_found"]:
        print(f"  Lead time: mean={stats['lead_time_hours_mean']:.1f} hr, "
              f"median={stats['lead_time_hours_median']:.1f} hr, "
              f"range=[{stats['lead_time_hours_min']:.1f}, "
              f"{stats['lead_time_hours_max']:.1f}] hr")
    if fault_breakdown:
        print(f"\n  Per fault type:")
        for fault, vals in sorted(fault_breakdown.items()):
            print(f"    {fault:<40} n={vals['n']:>2}  "
                  f"mean={vals['lead_time_hours_mean']:>6.1f} hr")

    print(f"\n[OK] Labels: {csv_path}")
    print(f"[OK] Stats:  {json_path}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--horizon-days", type=float, default=30.0,
                         help="Max look-back period for precursor search")
    args = parser.parse_args()

    all_stats = {}
    farms = ["A", "B", "C"] if args.farm == "all" else [args.farm]
    for farm in farms:
        all_stats[farm] = process_farm(farm, horizon_days=args.horizon_days)

    print("\n" + "=" * 80)
    print("  CARE-PRECURSOR SUMMARY")
    print("=" * 80)
    for farm, s in all_stats.items():
        if not s:
            continue
        print(f"  Farm {farm}: {s['n_precursor_found']}/{s['n_anomaly']} "
              f"anomaly events have a precursor alert within "
              f"{s['horizon_days']:.0f} days; "
              f"mean lead time = {s['lead_time_hours_mean']:.1f} hr "
              f"({s['lead_time_hours_mean']/24:.1f} days)")

    summary_path = OUT_DIR / "precursor_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n[OK] Cross-farm summary: {summary_path}")


if __name__ == "__main__":
    main()
