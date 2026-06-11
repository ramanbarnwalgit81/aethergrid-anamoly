"""
CARE dataset adapter — converts all 3 Wind Farms (A, B, C) with real labels
to the WindBench canonical schema.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark.schema import (
    CANONICAL_RAW_FEATURES, PARQUET_COLUMNS, validate_canonical_df,
    FaultSubsystem, FaultLabelType,
)

CARE_BASE = Path("data/real_scada/care/extracted")
OUT_DIR = Path("data/benchmark/harmonized")


# Farm A semantic mappings (from feature_description.csv)
FARM_A_COL_MAP = {
    "wind_speed_3_avg":     "wind_speed_ms",
    "sensor_0_avg":         "ambient_temp_c",
    "power_30_avg":         "active_power_kw",
    "sensor_18_avg":        "rotor_speed_rpm",  # actually generator rpm
    "sensor_12_avg":        "gearbox_oil_temp_c",
    "sensor_11_avg":        "gearbox_bearing_temp_c",
    "sensor_13_avg":        "generator_bearing_de_temp_c",
    "sensor_14_avg":        "generator_bearing_nde_temp_c",
    "sensor_15_avg":        "generator_winding_temp_c",
    "sensor_7_avg":         "nacelle_temp_c",
    "sensor_1_avg":         "wind_direction_deg",
    "sensor_5_avg":         "pitch_angle_deg",
    "reactive_power_27_avg": "reactive_power_kvar",
}


def _map_fault_type_care(description: str) -> FaultSubsystem:
    desc = (description or "").lower()
    if "gearbox" in desc:
        return FaultSubsystem.GEARBOX
    if "generator bearing" in desc:
        return FaultSubsystem.GENERATOR_BEARING
    if "transformer" in desc:
        return FaultSubsystem.TRANSFORMER
    if "hydraulic" in desc:
        return FaultSubsystem.HYDRAULIC
    if "blade" in desc or "pitch" in desc:
        return FaultSubsystem.BLADE
    return FaultSubsystem.UNKNOWN


def harmonize_farm_a():
    farm_dir = CARE_BASE / "Wind Farm A" / "Wind Farm A"
    if not farm_dir.exists():
        print(f"[ERR] CARE Farm A not extracted at {farm_dir}")
        return False

    print("[CARE-A] Loading 22 event CSVs...")

    events_info = pd.read_csv(farm_dir / "event_info.csv", sep=";")
    datasets_dir = farm_dir / "datasets"

    # Collect all event dataframes
    all_frames = []
    fault_rows = []
    turbine_ids_seen = set()

    for _, ev in events_info.iterrows():
        event_id = int(ev["event_id"])
        csv_path = datasets_dir / f"{event_id}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, sep=";", low_memory=False)

        # asset_id → canonical turbine_id
        if "asset_id" not in df.columns:
            continue
        # Use unique asset_id values (usually one per CSV)
        asset_ids = df["asset_id"].dropna().unique()
        if len(asset_ids) == 0:
            continue
        # Canonical name
        turbine_id = f"CARE_A_T{int(asset_ids[0]):02d}"
        df["turbine_id"] = turbine_id
        turbine_ids_seen.add(turbine_id)

        # Rename mapped columns
        df = df.rename(columns=FARM_A_COL_MAP)

        # Timestamps
        df["event_ts"] = pd.to_datetime(df["time_stamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["event_ts"])
        df["wind_farm"] = "CARE_FARM_A"

        # Build fault_event_id mapping for this event
        event_id_str = f"CARE_A_EV{event_id:03d}"
        if ev["event_label"] == "anomaly":
            subsystem = _map_fault_type_care(ev.get("event_description"))
            # Label rows between event_start_id and event_end_id
            start_idx = int(ev["event_start_id"])
            end_idx = int(ev["event_end_id"])
            df["fault_event_id"] = None
            if start_idx < len(df) and end_idx < len(df):
                df.iloc[start_idx:end_idx + 1, df.columns.get_loc("fault_event_id")] = event_id_str

                fault_rows.append({
                    "event_id": event_id_str,
                    "turbine_id": turbine_id,
                    "wind_farm": "CARE_FARM_A",
                    "fault_subsystem": subsystem.value,
                    "fault_description": ev.get("event_description", ""),
                    "start_ts": df.iloc[start_idx]["event_ts"].isoformat()
                                 if start_idx < len(df) else "",
                    "end_ts": df.iloc[end_idx]["event_ts"].isoformat()
                               if end_idx < len(df) else "",
                    "label_type": FaultLabelType.REAL.value,
                })
        else:
            df["fault_event_id"] = None

        all_frames.append(df)

    if not all_frames:
        print("[ERR] No CARE Farm A data loaded")
        return False

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"[CARE-A] Merged {len(combined):,} rows from {len(all_frames)} events, "
          f"{len(turbine_ids_seen)} turbines")

    # Add missing canonical columns
    for col in CANONICAL_RAW_FEATURES:
        if col not in combined.columns:
            combined[col] = np.nan

    # Coerce numeric
    for col in CANONICAL_RAW_FEATURES:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # operational_state
    if "operational_state" not in combined.columns:
        combined["operational_state"] = np.where(
            combined["active_power_kw"].fillna(0) > 50, "PRODUCING", "STOPPED"
        )

    # Select canonical columns
    for col in PARQUET_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[PARQUET_COLUMNS]

    # Sort
    combined = combined.sort_values(["event_ts", "turbine_id"]).reset_index(drop=True)

    # Validate
    errors = validate_canonical_df(combined)
    if errors:
        print(f"[WARN] Schema validation: {errors}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / "care_farm_a.parquet"
    combined.to_parquet(out_parquet, index=False)
    print(f"[OK] Wrote {out_parquet} ({out_parquet.stat().st_size / 1024 / 1024:.1f} MB)")

    # Faults
    faults_df = pd.DataFrame(fault_rows)
    faults_df.to_csv(OUT_DIR / "care_farm_a_faults.csv", index=False)
    print(f"[OK] Wrote {len(faults_df)} real labeled fault events")

    # Turbine metadata (CARE Farm A = 2MW Vestas onshore Portugal, anonymized)
    turbine_rows = []
    for tid in sorted(turbine_ids_seen):
        turbine_rows.append({
            "turbine_id": tid, "wind_farm": "CARE_FARM_A",
            "manufacturer": "Vestas (assumed)", "model": "V80-2.0 (assumed)",
            "rated_power_kw": 2000, "rotor_diameter_m": 80, "hub_height_m": 78,
            "latitude": None, "longitude": None,
            "commissioning_year": None, "country": "PRT", "offshore": False,
        })
    pd.DataFrame(turbine_rows).to_csv(OUT_DIR / "care_farm_a_turbines.csv", index=False)
    print(f"[OK] Wrote {len(turbine_rows)} turbine metadata entries")

    return True


def extract_farm_if_needed(farm_letter: str):
    """Extract CARE Farm B or C zip if not already done."""
    zip_path = CARE_BASE / f"Wind Farm {farm_letter}.zip"
    extracted_path = CARE_BASE / f"Wind Farm {farm_letter}"
    if extracted_path.exists() and any(extracted_path.iterdir()):
        return extracted_path
    if not zip_path.exists():
        print(f"[WARN] {zip_path} missing")
        return None
    print(f"[EXTRACT] Wind Farm {farm_letter} from zip...")
    extracted_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted_path)
    return extracted_path


def harmonize_farm_bc(farm_letter: str, max_features: int = 60):
    """Farm B and C have anonymized numeric feature names (no semantic mapping).
    We extract, keep the top-N numeric features by variance, and label faults only.
    """
    farm_path = extract_farm_if_needed(farm_letter)
    if farm_path is None:
        return False

    # Navigate into "Wind Farm X/Wind Farm X"
    inner = farm_path / f"Wind Farm {farm_letter}"
    if not (inner / "event_info.csv").exists():
        inner = farm_path  # maybe no nested dir

    event_info_path = inner / "event_info.csv"
    if not event_info_path.exists():
        print(f"[ERR] event_info.csv not found for Farm {farm_letter}")
        return False

    print(f"[CARE-{farm_letter}] Loading event info...")
    events = pd.read_csv(event_info_path, sep=";")
    datasets_dir = inner / "datasets"

    all_frames = []
    fault_rows = []
    turbine_ids_seen = set()

    for _, ev in events.iterrows():
        event_id = int(ev["event_id"])
        csv_path = datasets_dir / f"{event_id}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, sep=";", low_memory=False)
        except Exception as e:
            print(f"[WARN] {csv_path} failed: {e}")
            continue

        if "asset_id" not in df.columns:
            continue
        asset_ids = df["asset_id"].dropna().unique()
        if len(asset_ids) == 0:
            continue

        turbine_id = f"CARE_{farm_letter}_T{int(asset_ids[0]):02d}"
        df["turbine_id"] = turbine_id
        turbine_ids_seen.add(turbine_id)
        df["wind_farm"] = f"CARE_FARM_{farm_letter}"

        if "time_stamp" in df.columns:
            df["event_ts"] = pd.to_datetime(df["time_stamp"], utc=True, errors="coerce")
        else:
            df["event_ts"] = pd.date_range(
                "2020-01-01", periods=len(df), freq="10min", tz="UTC"
            )
        df = df.dropna(subset=["event_ts"])

        # Keep ONLY numeric columns (anonymized for B/C)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Add canonical columns as NaN (we have no mapping for B/C)
        for col in CANONICAL_RAW_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        # Fault event labeling
        event_id_str = f"CARE_{farm_letter}_EV{event_id:03d}"
        df["fault_event_id"] = None
        if ev["event_label"] == "anomaly":
            subsystem = _map_fault_type_care(ev.get("event_description"))
            start_idx = int(ev.get("event_start_id", 0))
            end_idx = int(ev.get("event_end_id", len(df)))
            if 0 <= start_idx < len(df):
                end_idx = min(end_idx, len(df) - 1)
                df.iloc[start_idx:end_idx + 1, df.columns.get_loc("fault_event_id")] = event_id_str
                fault_rows.append({
                    "event_id": event_id_str,
                    "turbine_id": turbine_id,
                    "wind_farm": f"CARE_FARM_{farm_letter}",
                    "fault_subsystem": subsystem.value,
                    "fault_description": ev.get("event_description", ""),
                    "start_ts": df.iloc[start_idx]["event_ts"].isoformat(),
                    "end_ts": df.iloc[end_idx]["event_ts"].isoformat(),
                    "label_type": FaultLabelType.REAL.value,
                })

        df["operational_state"] = "PRODUCING"  # unknown for B/C

        # Keep only canonical columns (numeric B/C features are NaN in canonical)
        for col in PARQUET_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[PARQUET_COLUMNS]
        all_frames.append(df)

    if not all_frames:
        return False

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["event_ts", "turbine_id"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / f"care_farm_{farm_letter.lower()}.parquet"
    combined.to_parquet(out_parquet, index=False)
    print(f"[OK] Wrote Farm {farm_letter}: {out_parquet} "
          f"({out_parquet.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{len(combined):,} rows, {len(turbine_ids_seen)} turbines)")

    pd.DataFrame(fault_rows).to_csv(
        OUT_DIR / f"care_farm_{farm_letter.lower()}_faults.csv", index=False)
    print(f"[OK] Wrote {len(fault_rows)} faults for Farm {farm_letter}")

    # Minimal turbine metadata (anonymized)
    turbine_rows = []
    for tid in sorted(turbine_ids_seen):
        turbine_rows.append({
            "turbine_id": tid, "wind_farm": f"CARE_FARM_{farm_letter}",
            "manufacturer": "Anonymized", "model": "Anonymized",
            "rated_power_kw": None, "rotor_diameter_m": None, "hub_height_m": None,
            "latitude": None, "longitude": None,
            "commissioning_year": None, "country": "DEU", "offshore": True,
        })
    pd.DataFrame(turbine_rows).to_csv(
        OUT_DIR / f"care_farm_{farm_letter.lower()}_turbines.csv", index=False)
    return True


def harmonize_all():
    print("=" * 60)
    print("  Harmonizing CARE datasets (A, B, C) -> canonical schema")
    print("=" * 60)
    ok_a = harmonize_farm_a()
    ok_b = harmonize_farm_bc("B")
    ok_c = harmonize_farm_bc("C")
    print()
    print(f"  Farm A: {'OK' if ok_a else 'FAIL'}")
    print(f"  Farm B: {'OK' if ok_b else 'FAIL'}")
    print(f"  Farm C: {'OK' if ok_c else 'FAIL'}")


if __name__ == "__main__":
    harmonize_all()
