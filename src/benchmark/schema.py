"""
WindBench Canonical Schema.

Defines a unified data model that ALL source datasets (Kelmarsh, Penmanshiel,
CARE, EDP, etc.) are converted to. This is the foundation of the benchmark:
every evaluation, every model, every metric operates on this single schema.

Schema version: 1.0
Target format: Apache Parquet (columnar, compressed, fast to load)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional


# ═══════════════════════════════════════════════════════════════════
# CANONICAL FEATURES (cross-dataset standard)
# ═══════════════════════════════════════════════════════════════════
# Every dataset must map its columns to these names, or mark them as
# missing. This guarantees cross-dataset model compatibility.

CANONICAL_RAW_FEATURES = [
    # Environmental (required)
    "wind_speed_ms",
    "ambient_temp_c",
    # Electrical (required)
    "active_power_kw",
    # Mechanical — rotor/drivetrain (required)
    "rotor_speed_rpm",
    # Thermal — drivetrain (at least one required)
    "gearbox_oil_temp_c",
    "gearbox_bearing_temp_c",
    "main_bearing_temp_c",
    # Thermal — generator (optional)
    "generator_bearing_de_temp_c",
    "generator_bearing_nde_temp_c",
    "generator_winding_temp_c",
    # Thermal — nacelle (optional)
    "nacelle_temp_c",
    # Vibration (optional — not all farms instrument)
    "vibration_x_rms",
    "vibration_y_rms",
    # Meteorological (optional)
    "wind_direction_deg",
    "pressure_pa",
    "humidity_pct",
    # Electrical detail (optional)
    "reactive_power_kvar",
    "generator_rpm",
    "grid_frequency_hz",
    "pitch_angle_deg",
]

# Minimum set a dataset must have to be usable
REQUIRED_FEATURES = [
    "wind_speed_ms",
    "ambient_temp_c",
    "active_power_kw",
    "rotor_speed_rpm",
]

# At least ONE of these must be present (drivetrain temperature indicator)
AT_LEAST_ONE_OF = [
    "gearbox_oil_temp_c",
    "gearbox_bearing_temp_c",
    "main_bearing_temp_c",
]


class FaultLabelType(Enum):
    SYNTHETIC = "synthetic"      # Programmatically injected
    REAL = "real"                # From maintenance logbook
    MIXED = "mixed"              # Combination (not recommended)


class FaultSubsystem(Enum):
    """Standardized fault subsystem taxonomy."""
    GEARBOX = "gearbox"
    MAIN_BEARING = "main_bearing"
    GENERATOR = "generator"
    GENERATOR_BEARING = "generator_bearing"
    BLADE = "blade"
    PITCH_SYSTEM = "pitch_system"
    YAW_SYSTEM = "yaw_system"
    HYDRAULIC = "hydraulic"
    ELECTRICAL = "electrical"
    TRANSFORMER = "transformer"
    CONVERTER = "converter"
    CONTROL_SYSTEM = "control_system"
    SENSOR = "sensor"
    UNKNOWN = "unknown"


@dataclass
class TurbineMeta:
    """Per-turbine metadata (one row in dim_turbine)."""
    turbine_id: str                  # Canonical ID e.g., "KELMARSH_T01"
    wind_farm: str                   # Canonical farm name
    manufacturer: str                # e.g., "Senvion", "Vestas", "Siemens"
    model: str                       # e.g., "MM92", "V90-2.0"
    rated_power_kw: float
    rotor_diameter_m: float
    hub_height_m: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    commissioning_year: Optional[int] = None
    country: Optional[str] = None    # ISO alpha-3 ("GBR", "DEU", "USA")
    offshore: bool = False


@dataclass
class FaultEvent:
    """A labeled fault event with start/end timestamps."""
    event_id: str
    turbine_id: str
    wind_farm: str
    fault_subsystem: FaultSubsystem
    fault_description: str
    start_ts: str                    # ISO-8601 UTC
    end_ts: str
    label_type: FaultLabelType
    severity: Optional[str] = None   # "critical" / "warning" / "advisory"
    repair_cost_usd: Optional[float] = None
    downtime_hours: Optional[float] = None


@dataclass
class DatasetManifest:
    """Top-level descriptor for a dataset in the benchmark."""
    dataset_id: str                  # e.g., "KELMARSH_2016_2018"
    version: str                     # semantic version, e.g., "1.0.0"
    source_url: str
    license: str                     # e.g., "CC-BY-4.0"
    citation: str                    # Preferred citation string
    n_turbines: int
    n_rows: int
    n_faults: int
    fault_label_type: FaultLabelType
    date_min: str
    date_max: str
    resolution_seconds: int          # e.g., 3600 for hourly
    canonical_parquet_path: str
    turbine_metadata_csv: str
    fault_events_csv: str
    notes: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# CANONICAL PARQUET COLUMN ORDER
# ═══════════════════════════════════════════════════════════════════

PARQUET_COLUMNS = (
    ["event_ts", "turbine_id", "wind_farm"]
    + CANONICAL_RAW_FEATURES
    + ["operational_state", "fault_event_id"]  # operational_state: "PRODUCING"/"STOPPED"/"MAINTENANCE"
)


def validate_canonical_df(df) -> List[str]:
    """Return list of schema violations (empty = valid)."""
    errors = []

    # Required columns
    for col in ["event_ts", "turbine_id", "wind_farm"]:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            errors.append(f"Missing required feature: {col}")

    # At least one drivetrain temperature
    has_any = any(c in df.columns for c in AT_LEAST_ONE_OF)
    if not has_any:
        errors.append(f"Must have at least one of: {AT_LEAST_ONE_OF}")

    # Event_ts dtype
    if "event_ts" in df.columns:
        import pandas as pd
        if not pd.api.types.is_datetime64_any_dtype(df["event_ts"]):
            errors.append(f"event_ts must be datetime64, got {df['event_ts'].dtype}")

    return errors
