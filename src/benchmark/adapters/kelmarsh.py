"""
Kelmarsh dataset adapter — converts existing processed Kelmarsh parquet
to WindBench canonical schema.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.benchmark.schema import (
    CANONICAL_RAW_FEATURES, PARQUET_COLUMNS, validate_canonical_df,
    FaultSubsystem, FaultLabelType,
)

SOURCE_PARQUET = Path("data/processed/scada_full.parquet")
OUT_DIR = Path("data/benchmark/harmonized")


KELMARSH_TURBINES = [
    ("KELMARSH_T01", "KELMARSH", "Senvion", "MM92", 2000, 92, 80, 52.3972, -0.9404, 2016, "GBR", False),
    ("KELMARSH_T02", "KELMARSH", "Senvion", "MM92", 2000, 92, 80, 52.3975, -0.9410, 2016, "GBR", False),
    ("KELMARSH_T03", "KELMARSH", "Senvion", "MM92", 2000, 92, 80, 52.3979, -0.9415, 2016, "GBR", False),
    ("KELMARSH_T04", "KELMARSH", "Senvion", "MM92", 2000, 92, 80, 52.3983, -0.9420, 2016, "GBR", False),
    ("KELMARSH_T05", "KELMARSH", "Senvion", "MM92", 2000, 92, 80, 52.3987, -0.9425, 2016, "GBR", False),
    ("KELMARSH_T06", "KELMARSH", "Senvion", "MM92", 2000, 92, 80, 52.3991, -0.9430, 2016, "GBR", False),
]


def _map_turbine_id(legacy: str) -> str:
    """KWF-001 -> KELMARSH_T01"""
    num = int(legacy.split("-")[1])
    return f"KELMARSH_T{num:02d}"


def _map_fault_type(injected: str) -> tuple:
    """Map legacy fault name to canonical subsystem."""
    if not injected or pd.isna(injected):
        return None, None
    if "gearbox" in injected:
        return FaultSubsystem.GEARBOX, "Gearbox temperature spike"
    if "bearing" in injected:
        return FaultSubsystem.MAIN_BEARING, "Main bearing overheat"
    if "vibration" in injected:
        return FaultSubsystem.BLADE, "Drivetrain vibration anomaly"
    return FaultSubsystem.UNKNOWN, injected


def harmonize():
    if not SOURCE_PARQUET.exists():
        print(f"[ERR] {SOURCE_PARQUET} missing — run Kelmarsh loader first.")
        return False

    print(f"[KELMARSH] Loading {SOURCE_PARQUET}...")
    df = pd.read_parquet(SOURCE_PARQUET)
    print(f"[KELMARSH] {len(df):,} rows")

    # Map columns to canonical names
    rename = {
        "gearbox_temp_c": "gearbox_oil_temp_c",
        "main_bearing_temp_c": "main_bearing_temp_c",
    }
    df = df.rename(columns=rename)

    # Map turbine IDs
    df["turbine_id_canonical"] = df["turbine_id"].apply(_map_turbine_id)
    df["wind_farm"] = "KELMARSH"
    df["turbine_id"] = df["turbine_id_canonical"]
    df = df.drop(columns=["turbine_id_canonical"])

    # Add missing optional columns as NaN
    for col in CANONICAL_RAW_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Build fault events table
    fault_rows = []
    for (tid, ftype), group in df[df["injected_fault"].notna()].groupby(
            ["turbine_id", "injected_fault"]):
        subsystem, desc = _map_fault_type(ftype)
        fault_rows.append({
            "event_id": f"KELMARSH_{tid}_{ftype}_{int(group['event_ts'].iloc[0].timestamp())}",
            "turbine_id": tid,
            "wind_farm": "KELMARSH",
            "fault_subsystem": subsystem.value if subsystem else "unknown",
            "fault_description": desc,
            "start_ts": group["event_ts"].min().isoformat(),
            "end_ts": group["event_ts"].max().isoformat(),
            "label_type": FaultLabelType.SYNTHETIC.value,
        })
    faults_df = pd.DataFrame(fault_rows)

    # Add fault_event_id column to main table (null for normal rows)
    event_id_map = {(r["turbine_id"], r["fault_subsystem"]): r["event_id"]
                    for r in fault_rows}
    def get_event_id(row):
        if pd.isna(row["injected_fault"]):
            return None
        subsystem, _ = _map_fault_type(row["injected_fault"])
        return event_id_map.get((row["turbine_id"], subsystem.value if subsystem else None))
    df["fault_event_id"] = df.apply(get_event_id, axis=1)

    # operational_state
    if "operational_state" not in df.columns:
        df["operational_state"] = np.where(df["active_power_kw"].fillna(0) > 50,
                                             "PRODUCING", "STOPPED")

    # Final column selection
    final_cols = [c for c in PARQUET_COLUMNS if c in df.columns]
    for col in PARQUET_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[PARQUET_COLUMNS]

    # Validate
    errors = validate_canonical_df(df)
    if errors:
        print(f"[WARN] Schema validation errors: {errors}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / "kelmarsh.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"[OK] Wrote canonical parquet: {out_parquet}")

    faults_df.to_csv(OUT_DIR / "kelmarsh_faults.csv", index=False)
    print(f"[OK] Wrote {len(faults_df)} fault events")

    # Turbine metadata
    tmeta = pd.DataFrame(KELMARSH_TURBINES, columns=[
        "turbine_id", "wind_farm", "manufacturer", "model", "rated_power_kw",
        "rotor_diameter_m", "hub_height_m", "latitude", "longitude",
        "commissioning_year", "country", "offshore",
    ])
    tmeta.to_csv(OUT_DIR / "kelmarsh_turbines.csv", index=False)
    print(f"[OK] Wrote turbine metadata")

    return True


if __name__ == "__main__":
    harmonize()
