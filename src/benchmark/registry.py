"""
WindBench Dataset Registry.

Central manifest of all benchmark datasets. New datasets are added here;
each must provide an adapter that converts source format to canonical schema.
"""

from pathlib import Path
from src.benchmark.schema import DatasetManifest, FaultLabelType

BENCHMARK_DIR = Path("data/benchmark")
HARMONIZED_DIR = BENCHMARK_DIR / "harmonized"


REGISTRY = {
    "KELMARSH": DatasetManifest(
        dataset_id="KELMARSH",
        version="1.0.0",
        source_url="https://zenodo.org/records/5841834",
        license="CC-BY-4.0",
        citation=("Cubico Sustainable Investments, 'Kelmarsh Wind Farm Data', "
                  "Zenodo 5841834, CC-BY-4.0, 2022."),
        n_turbines=6,
        n_rows=79833,
        n_faults=4,
        fault_label_type=FaultLabelType.SYNTHETIC,
        date_min="2016-01-03T00:00:00+00:00",
        date_max="2018-12-31T23:00:00+00:00",
        resolution_seconds=3600,
        canonical_parquet_path=str(HARMONIZED_DIR / "kelmarsh.parquet"),
        turbine_metadata_csv=str(HARMONIZED_DIR / "kelmarsh_turbines.csv"),
        fault_events_csv=str(HARMONIZED_DIR / "kelmarsh_faults.csv"),
        notes=("Senvion MM92 2MW, Northamptonshire UK. "
               "Vibration synthesized from RPM. Faults injected in last 7 days."),
    ),

    "PENMANSHIEL": DatasetManifest(
        dataset_id="PENMANSHIEL",
        version="1.0.0",
        source_url="https://zenodo.org/records/5946808",
        license="CC-BY-4.0",
        citation=("Cubico Sustainable Investments, 'Penmanshiel Wind Farm Data', "
                  "Zenodo 5946808, CC-BY-4.0, 2022."),
        n_turbines=5,
        n_rows=21473,
        n_faults=2,
        fault_label_type=FaultLabelType.SYNTHETIC,
        date_min="2021-01-01T00:00:00+00:00",
        date_max="2021-07-01T00:00:00+00:00",
        resolution_seconds=3600,
        canonical_parquet_path=str(HARMONIZED_DIR / "penmanshiel.parquet"),
        turbine_metadata_csv=str(HARMONIZED_DIR / "penmanshiel_turbines.csv"),
        fault_events_csv=str(HARMONIZED_DIR / "penmanshiel_faults.csv"),
        notes=("Senvion MM82 2.05MW, Scottish Borders UK. "
               "WT11-15 (5 turbines). 6-month window."),
    ),

    "CARE_FARM_A": DatasetManifest(
        dataset_id="CARE_FARM_A",
        version="1.0.0",
        source_url="https://zenodo.org/records/14006163",
        license="CC-BY-SA-4.0",
        citation=("Kreutz, M., et al., 'CARE to Compare: A Real-World Benchmark "
                  "Dataset for Early Fault Detection in Wind Turbine Data', "
                  "Data 9(12):138, 2024."),
        n_turbines=5,
        n_rows=1_190_000,  # approx across all events
        n_faults=11,
        fault_label_type=FaultLabelType.REAL,
        date_min="2014-01-01T00:00:00+00:00",
        date_max="2023-12-31T23:59:59+00:00",
        resolution_seconds=600,
        canonical_parquet_path=str(HARMONIZED_DIR / "care_farm_a.parquet"),
        turbine_metadata_csv=str(HARMONIZED_DIR / "care_farm_a_turbines.csv"),
        fault_events_csv=str(HARMONIZED_DIR / "care_farm_a_faults.csv"),
        notes=("REAL LABELED FAULTS. 2MW Vestas onshore Portugal "
               "(EDP open data). 22 events (11 anomaly + 11 normal). "
               "Fault types: Gearbox, Generator bearing, Hydraulic group, Transformer."),
    ),

    "CARE_FARM_B": DatasetManifest(
        dataset_id="CARE_FARM_B",
        version="1.0.0",
        source_url="https://zenodo.org/records/14006163",
        license="CC-BY-SA-4.0",
        citation=("Kreutz, M., et al., 'CARE to Compare', Data 9(12):138, 2024."),
        n_turbines=0,  # filled by adapter
        n_rows=0,
        n_faults=0,
        fault_label_type=FaultLabelType.REAL,
        date_min="",
        date_max="",
        resolution_seconds=600,
        canonical_parquet_path=str(HARMONIZED_DIR / "care_farm_b.parquet"),
        turbine_metadata_csv=str(HARMONIZED_DIR / "care_farm_b_turbines.csv"),
        fault_events_csv=str(HARMONIZED_DIR / "care_farm_b_faults.csv"),
        notes="German offshore, 257 features, anonymized.",
    ),

    "CARE_FARM_C": DatasetManifest(
        dataset_id="CARE_FARM_C",
        version="1.0.0",
        source_url="https://zenodo.org/records/14006163",
        license="CC-BY-SA-4.0",
        citation=("Kreutz, M., et al., 'CARE to Compare', Data 9(12):138, 2024."),
        n_turbines=0,
        n_rows=0,
        n_faults=0,
        fault_label_type=FaultLabelType.REAL,
        date_min="",
        date_max="",
        resolution_seconds=600,
        canonical_parquet_path=str(HARMONIZED_DIR / "care_farm_c.parquet"),
        turbine_metadata_csv=str(HARMONIZED_DIR / "care_farm_c_turbines.csv"),
        fault_events_csv=str(HARMONIZED_DIR / "care_farm_c_faults.csv"),
        notes="German offshore, 957 features, anonymized.",
    ),

    "EDP_OPENDATA": DatasetManifest(
        dataset_id="EDP_OPENDATA",
        version="1.0.0",
        source_url="https://www.edp.com/en/innovation/open-data",
        license="EDP Open Data License",
        citation=("EDP, 'Open Data Portal — Wind Farm Operational Data', 2016-2017."),
        n_turbines=0,  # 5 turbines from T01-T11
        n_rows=0,
        n_faults=0,
        fault_label_type=FaultLabelType.REAL,
        date_min="2016-01-01T00:00:00+00:00",
        date_max="2017-12-31T23:59:59+00:00",
        resolution_seconds=600,
        canonical_parquet_path=str(HARMONIZED_DIR / "edp_opendata.parquet"),
        turbine_metadata_csv=str(HARMONIZED_DIR / "edp_opendata_turbines.csv"),
        fault_events_csv=str(HARMONIZED_DIR / "edp_opendata_faults.csv"),
        notes=("2MW Vestas onshore Portugal. 10-min SCADA + failures logbook."),
    ),
}


def get_manifest(dataset_id: str) -> DatasetManifest:
    if dataset_id not in REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_id}. "
                          f"Available: {list(REGISTRY.keys())}")
    return REGISTRY[dataset_id]


def list_available() -> list:
    """Return list of dataset_ids whose canonical parquet exists on disk."""
    return [did for did, m in REGISTRY.items()
            if Path(m.canonical_parquet_path).exists()]


def list_pending() -> list:
    """Return dataset_ids not yet harmonized."""
    return [did for did, m in REGISTRY.items()
            if not Path(m.canonical_parquet_path).exists()]
