"""CARE-Precursor sensitivity analysis: sweep (sustained-window, horizon).

For each of the 11 CARE Farm A anomaly events, sweep the two hyperparameters
of the precursor definition:
  - sustained window length: number of consecutive non-zero status_type_id
    rows required to qualify as a "sustained" operator alert (paper default: 12 rows = 2 h)
  - look-back horizon (days): max days before event_start_id to search for a
    sustained alert (paper default: 30 days)

For each (window, horizon) pair we count the number of events that have a
precursor and report the median lead time. This quantifies whether the
"23% of events have precursors" headline is robust to the threshold choice.

Usage:  python -m src.benchmark.precursor_sensitivity
Outputs: docs/results/precursor_sensitivity.json
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

RESULTS_DIR = Path("docs/results")
CARE_DIR = Path("data/real_scada/care/extracted/Wind Farm A/Wind Farm A")
RES_MINUTES = 10


def find_precursor(df: pd.DataFrame, event_start_id: int,
                     horizon_rows: int, window_rows: int) -> tuple[int | None, int | None]:
    """Return (precursor_start_id, lead_time_rows) or (None, None).

    Mirrors care_precursor.find_precursor: find the LAST sustained-normal run
    of length >= window_rows inside the horizon; the precursor starts one row
    after that run. If the turbine stayed in normal status up to event_start_id
    there is no precursor (return None).
    """
    start = max(0, event_start_id - horizon_rows)
    window = df.iloc[start:event_start_id]
    if "status_type_id" not in df.columns or len(window) < window_rows:
        return None, None
    status = window["status_type_id"].fillna(0).astype(int).to_numpy()
    is_normal = (status == 0).astype(int)

    n = len(is_normal)
    last_end = -1
    run = 0
    for i in range(n):
        if is_normal[i] == 1:
            run += 1
        else:
            if run >= window_rows:
                last_end = i - 1
            run = 0
    if run >= window_rows:
        last_end = n - 1  # turbine normal through end of window
    if last_end == -1 or last_end >= n - 1:
        return None, None
    pre_idx = start + last_end + 1
    return pre_idx, event_start_id - pre_idx


def sweep_farm_A():
    event_info = pd.read_csv(CARE_DIR / "event_info.csv", sep=";")
    anomaly_events = event_info[event_info["event_label"] == "anomaly"]
    # Sweep grids
    window_hours = [1, 2, 3, 6, 12]       # sustained-alert duration (hours)
    horizon_days_grid = [7, 14, 30, 60, 90]

    rows_per_hour = 60 // RES_MINUTES       # 6

    grid = {}
    for wh in window_hours:
        w_rows = wh * rows_per_hour
        for hd in horizon_days_grid:
            h_rows = int(hd * 24 * rows_per_hour)
            found_count = 0
            lead_times = []
            for _, ev in anomaly_events.iterrows():
                csv = CARE_DIR / "datasets" / f"{int(ev['event_id'])}.csv"
                if not csv.exists():
                    continue
                df = pd.read_csv(csv, sep=";", low_memory=False)
                pre_id, lead_rows = find_precursor(
                    df, int(ev["event_start_id"]), h_rows, w_rows)
                if pre_id is not None:
                    found_count += 1
                    lead_times.append(lead_rows / rows_per_hour)    # hours
            grid[f"w{wh}h_h{hd}d"] = {
                "sustained_window_hours": wh,
                "horizon_days": hd,
                "events_with_precursor": found_count,
                "median_lead_time_hours": float(np.median(lead_times)) if lead_times else None,
                "max_lead_time_hours": float(np.max(lead_times)) if lead_times else None,
            }
    return grid


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": "CARE Farm A",
        "n_anomaly_events": 11,
        "paper_default": {"sustained_window_hours": 2, "horizon_days": 30},
        "grid": sweep_farm_A(),
    }
    with (RESULTS_DIR / "precursor_sensitivity.json").open("w") as f:
        json.dump(out, f, indent=2, default=str)
    # Print summary heatmap
    print(f"{'':>8}", end="")
    horizon_days_grid = [7, 14, 30, 60, 90]
    for hd in horizon_days_grid:
        print(f" h={hd:>3}d", end="")
    print()
    window_hours = [1, 2, 3, 6, 12]
    for wh in window_hours:
        print(f"w={wh:>2}h | ", end="")
        for hd in horizon_days_grid:
            key = f"w{wh}h_h{hd}d"
            n = out["grid"][key]["events_with_precursor"]
            print(f"   {n:>2}/11", end="")
        print()
    print("\n[OK] Saved: docs/results/precursor_sensitivity.json")


if __name__ == "__main__":
    main()
