"""
Penmanshiel dataset adapter — converts Greenbyte-export SCADA CSVs
(2016-2021, 14 Senvion MM82 turbines) to WindBench canonical schema.

Greenbyte format has 9 # comment lines at the top; the last comment line
is actually the column header (also prefixed with #). No fault labels are
provided, so this dataset is used for NORMAL distribution learning and
domain transfer studies.
"""

from pathlib import Path
import csv
import io
import re

import pandas as pd
import numpy as np

from src.benchmark.schema import CANONICAL_RAW_FEATURES

SOURCE_BASE = Path("data/real_scada/penmanshiel/extracted")
OUT_DIR = Path("data/benchmark/harmonized")


# Penmanshiel Greenbyte column -> canonical feature name
COL_MAP = {
    "Wind speed (m/s)": "wind_speed_ms",
    "Nacelle ambient temperature (°C)": "ambient_temp_c",
    "Power (kW)": "active_power_kw",
    "Rotor speed (RPM)": "rotor_speed_rpm",
    "Generator RPM (RPM)": "generator_rpm",
    "Front bearing temperature (°C)": "main_bearing_temp_c",
    "Generator bearing front temperature (°C)": "generator_bearing_de_temp_c",
    "Generator bearing rear temperature (°C)": "generator_bearing_nde_temp_c",
    "Nacelle temperature (°C)": "nacelle_temp_c",
    "Grid frequency (Hz)": "grid_frequency_hz",
    "Reactive power (kvar)": "reactive_power_kvar",
    "Wind direction (°)": "wind_direction_deg",
}


def _read_greenbyte_csv(csv_path: Path) -> pd.DataFrame:
    """Parse Greenbyte CSV: skip # comments, extract header from last #-line."""
    comment_lines = []
    with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                comment_lines.append(line.lstrip("#").strip())
                continue
            break

    if not comment_lines:
        return pd.DataFrame()

    # Last comment line should be the header (starts with "Date and time")
    header_line = None
    for ln in reversed(comment_lines):
        if "Date and time" in ln or "date" in ln.lower():
            header_line = ln
            break
    if header_line is None:
        header_line = comment_lines[-1]

    # Parse CSV header respecting quoting
    reader = csv.reader(io.StringIO(header_line))
    cols = next(reader)
    # Clean mojibake-encoded degrees
    cols = [c.replace("�", "°") for c in cols]

    # Read data without header
    df = pd.read_csv(csv_path, skiprows=len(comment_lines),
                      header=None, names=cols, low_memory=False)
    return df


def _extract_turbine_id(filename: str) -> str:
    """Turbine_Data_Penmanshiel_01_2020-...csv -> PENMANSHIEL_T01."""
    m = re.search(r"Penmanshiel_(\d+)_", filename)
    if m:
        return f"PENMANSHIEL_T{int(m.group(1)):02d}"
    return "PENMANSHIEL_UNKNOWN"


def _extract_year(filename: str) -> int:
    m = re.search(r"_(20\d{2})-\d{2}-\d{2}_", filename)
    return int(m.group(1)) if m else 0


def harmonize(years: list = None, max_per_year: int = None) -> bool:
    """
    Harmonize Penmanshiel CSVs into a single canonical parquet.

    Args:
        years: list of years to include (default: all found).
        max_per_year: optional cap on turbines per year (for speed).
    """
    if not SOURCE_BASE.exists():
        print(f"[ERR] Penmanshiel extracted data not found at {SOURCE_BASE}")
        return False

    if years is None:
        years = sorted([int(d.name) for d in SOURCE_BASE.iterdir()
                         if d.is_dir() and d.name.isdigit()])

    print(f"[PENMANSHIEL] Harmonizing years: {years}")

    all_frames = []
    turbine_ids_seen = set()
    for yr in years:
        year_dir = SOURCE_BASE / str(yr)
        if not year_dir.exists():
            continue
        csvs = sorted(year_dir.glob("Turbine_Data_Penmanshiel_*.csv"))
        if max_per_year:
            csvs = csvs[:max_per_year]
        print(f"  [{yr}] {len(csvs)} turbine CSVs")

        for i, csv_path in enumerate(csvs):
            try:
                df = _read_greenbyte_csv(csv_path)
            except Exception as e:
                print(f"    [SKIP] {csv_path.name}: {e}")
                continue
            if df.empty:
                continue

            tid = _extract_turbine_id(csv_path.name)
            turbine_ids_seen.add(tid)

            # First column is timestamp
            ts_col = df.columns[0]
            df["event_ts"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

            # Rename canonical columns
            rename_dict = {src: dst for src, dst in COL_MAP.items()
                           if src in df.columns}
            df = df.rename(columns=rename_dict)

            # Keep only event_ts + canonical features + turbine_id
            canonical_present = [c for c in CANONICAL_RAW_FEATURES if c in df.columns]
            keep = ["event_ts"] + canonical_present
            df = df[keep].copy()
            df["turbine_id"] = tid
            df["wind_farm"] = "PENMANSHIEL"

            # Coerce numeric columns
            for c in canonical_present:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=["event_ts"])
            all_frames.append(df)

            if (i + 1) % 5 == 0:
                print(f"    [{yr}] {i+1}/{len(csvs)} processed, "
                      f"total rows so far: {sum(len(f) for f in all_frames):,}")

    if not all_frames:
        print("[ERR] No data loaded")
        return False

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["turbine_id", "event_ts"]).reset_index(drop=True)

    # Fill missing canonical cols with NaN
    for col in CANONICAL_RAW_FEATURES:
        if col not in combined.columns:
            combined[col] = np.nan

    # Write parquet
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "penmanshiel_full.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\n[OK] Wrote {out_path}")
    print(f"     Rows: {len(combined):,}")
    print(f"     Turbines: {len(turbine_ids_seen)} ({sorted(turbine_ids_seen)})")
    print(f"     Date range: {combined['event_ts'].min()} to {combined['event_ts'].max()}")

    # Write turbine metadata
    turbines_df = pd.DataFrame({
        "turbine_id": sorted(turbine_ids_seen),
        "wind_farm": "PENMANSHIEL",
        "manufacturer": "Senvion",
        "model": "MM82",
        "rated_power_kw": 2050,
        "rotor_diameter_m": 82,
        "hub_height_m": 59,
        "country": "GBR",
    })
    turbines_df.to_csv(OUT_DIR / "penmanshiel_full_turbines.csv", index=False)
    print(f"[OK] Wrote {OUT_DIR / 'penmanshiel_full_turbines.csv'}")

    return True


if __name__ == "__main__":
    harmonize()
