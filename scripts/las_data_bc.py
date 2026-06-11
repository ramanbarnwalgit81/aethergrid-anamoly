"""Compute LAS_data (Jaccard distance between event-window and status-code
positive-row sets) for CARE Farms B and C, over anomaly events only.

Label-only: uses ensemble_care_farm.load_event to build is_anomaly_event and
is_anomaly_status per event. No autoencoder training (so no OOM).

Run from repo root:  PYTHONPATH=. python scripts/las_data_bc.py
Output: docs/results/las_data_bc.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
CARE_BASE = Path("data/real_scada/care/extracted")


def las_data_farm(farm: str):
    """LAS_data over anomaly events, replicating load_event's label logic in
    pure pandas (no torch import, so it runs alongside the training jobs)."""
    farm_dir = CARE_BASE / f"Wind Farm {farm}" / f"Wind Farm {farm}"
    datasets_dir = farm_dir / "datasets"
    events = pd.read_csv(farm_dir / "event_info.csv", sep=";")
    js = []
    for _, row in events.iterrows():
        if row["event_label"] != "anomaly":
            continue
        eid = int(row["event_id"])
        csv = datasets_dir / f"{eid}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, sep=";", low_memory=False)
        n = len(df)
        status = (np.nan_to_num(df["status_type_id"].values, nan=0).astype(int)
                  if "status_type_id" in df.columns else np.zeros(n, int))
        is_status = status != 0
        is_event = np.zeros(n, dtype=bool)
        s = int(row["event_start_id"]); e = min(int(row["event_end_id"]), n - 1)
        is_event[s:e + 1] = True
        pe = set(np.where(is_event)[0]); pst = set(np.where(is_status)[0])
        if not (pe | pst):
            continue
        js.append(1.0 - len(pe & pst) / len(pe | pst))
    return len(js), (float(np.mean(js)) if js else None)


def main():
    out = {}
    for farm in ["B", "C"]:
        n, las = las_data_farm(farm)
        out[farm] = {"n_anomaly_events": n, "las_data": round(las, 4) if las is not None else None}
        print(f"Farm {farm}: anomaly events={n}  LAS_data={out[farm]['las_data']}")
    Path("docs/results/las_data_bc.json").write_text(json.dumps(out, indent=2))
    print("[OK] wrote docs/results/las_data_bc.json")


if __name__ == "__main__":
    main()
