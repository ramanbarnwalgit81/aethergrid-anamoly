"""
Reproducibility Statement Generator — IEEE Access paper asset.

Produces:
  1. A plain-text reproducibility / data-availability section
     (copy-paste into the paper's "Data Availability" section)
  2. A BibTeX file with all dataset references
  3. A JSON manifest listing code, data, and model artifacts

IEEE Access requires a Data Availability Statement in accepted papers.
This script generates it from the project's actual file structure.

Usage:
    python -m src.benchmark.reproducibility_statement

Output:
    docs/paper/data_availability_statement.txt
    docs/paper/dataset_references.bib
    docs/paper/reproducibility_manifest.json
"""

from __future__ import annotations
from pathlib import Path
import json, datetime, hashlib

OUT_DIR = Path("docs/paper")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry with DOIs and access URLs
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = [
    {
        "key": "care",
        "name": "CARE Wind Farm Dataset (Farms A, B, C)",
        "doi": "10.5281/zenodo.7814452",
        "url": "https://zenodo.org/record/7814452",
        "license": "CC BY 4.0",
        "turbine_years": 89,
        "farms": ["A", "B", "C"],
        "note": (
            "Used for all three findings. Two independent ground-truth signals: "
            "event_start_id (maintenance-log) and status_type_id (PLC status code). "
            "Farm A: 11 anomaly events. Farm B: 6. Farm C: 27."
        ),
        "bibtex": """\
@dataset{care2023,
  author       = {Kreutz, C. K. and others},
  title        = {{CARE}: A Benchmark Dataset for Wind Turbine
                  Condition Monitoring},
  year         = {2023},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7814452},
  url          = {https://zenodo.org/record/7814452},
  license      = {CC BY 4.0},
}""",
    },
    {
        "key": "kelmarsh",
        "name": "Kelmarsh Wind Farm SCADA Dataset",
        "doi": "10.5281/zenodo.5841834",
        "url": "https://zenodo.org/record/5841834",
        "license": "CC BY 4.0",
        "turbine_years": None,
        "note": "Used for cross-farm generalization validation (Finding 1).",
        "bibtex": """\
@dataset{kelmarsh2022,
  author       = {Plumley, C.},
  title        = {Kelmarsh Wind Farm Data},
  year         = {2022},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5841834},
  url          = {https://zenodo.org/record/5841834},
  license      = {CC BY 4.0},
}""",
    },
    {
        "key": "penmanshiel",
        "name": "Penmanshiel Wind Farm SCADA Dataset",
        "doi": "10.5281/zenodo.5946808",
        "url": "https://zenodo.org/record/5946808",
        "license": "CC BY 4.0",
        "turbine_years": None,
        "note": (
            "Used for Fleet-NBM pre-training (Finding 2). "
            "Normal-behaviour models trained on Penmanshiel and transferred to CARE."
        ),
        "bibtex": """\
@dataset{penmanshiel2022,
  author       = {Plumley, C.},
  title        = {Penmanshiel Wind Farm Data},
  year         = {2022},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5946808},
  url          = {https://zenodo.org/record/5946808},
  license      = {CC BY 4.0},
}""",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Code and model artifact registry
# ──────────────────────────────────────────────────────────────────────────────

CODE_ARTIFACTS = [
    {
        "file": "src/benchmark/label_ambiguity_test.py",
        "role": "Label Ambiguity Test (LAT) — Finding 1 diagnostic",
    },
    {
        "file": "src/benchmark/label_ambiguity_score.py",
        "role": "Label Ambiguity Score (LAS) formal module — IEEE-quality",
    },
    {
        "file": "src/benchmark/pinn_stacker.py",
        "role": "PINN-v2 XGBoost stacker — Finding 2 main experiment",
    },
    {
        "file": "src/benchmark/foundation_models.py",
        "role": "Chronos-T5-tiny benchmark — Finding 3 foundation-model comparison",
    },
    {
        "file": "src/benchmark/delong_bootstrap.py",
        "role": "Bootstrap significance tests (N_BOOT=2000) for all three findings",
    },
    {
        "file": "src/benchmark/crossfarm_v7_bootstrap.py",
        "role": "Cross-farm v7 AUC bootstrap — 44 anomaly events, 3 farms",
    },
    {
        "file": "src/benchmark/labeling_decision_tree.py",
        "role": "Practitioner decision flowchart figure (PNG + PDF)",
    },
]

RESULT_ARTIFACTS = [
    "docs/results/pinn_stacker_y_event.json",
    "docs/results/pinn_stacker_y_precursor.json",
    "docs/results/delong_bootstrap.json",
    "docs/results/label_ambiguity_test.json",
    "docs/results/las_per_farm_table.json",
    "docs/results/foundation_models_care_a.json",
    "docs/results/crossfarm_v7_bootstrap.json",
    "docs/results/care_farm_b_ensemble.json",
    "docs/results/care_farm_c_ensemble.json",
]


# ──────────────────────────────────────────────────────────────────────────────
# Statement generator
# ──────────────────────────────────────────────────────────────────────────────

def _sha256_short(path: Path) -> str:
    if not path.exists():
        return "file-not-found"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def generate_data_availability_statement() -> str:
    today = datetime.date.today().isoformat()
    lines = [
        "DATA AVAILABILITY STATEMENT",
        "=" * 60,
        "",
        "All datasets used in this study are publicly available:",
        "",
    ]
    for i, ds in enumerate(DATASETS, 1):
        lines.append(f"  [{i}] {ds['name']}")
        lines.append(f"      DOI:     {ds['doi']}")
        lines.append(f"      URL:     {ds['url']}")
        lines.append(f"      License: {ds['license']}")
        lines.append(f"      Role:    {ds['note']}")
        lines.append("")

    lines += [
        "CODE AVAILABILITY",
        "-" * 60,
        "All experimental code is available at:",
        "  [Repository URL — add GitHub/Zenodo link before submission]",
        "",
        "Key scripts and their roles:",
    ]
    for art in CODE_ARTIFACTS:
        lines.append(f"  - {art['file']}")
        lines.append(f"      {art['role']}")

    lines += [
        "",
        "RESULT FILES",
        "-" * 60,
        "Pre-computed result JSONs are included in the repository under docs/results/:",
    ]
    for rf in RESULT_ARTIFACTS:
        p = Path(rf)
        sha = _sha256_short(p)
        lines.append(f"  - {rf}  [sha256:{sha}]")

    lines += [
        "",
        "REPRODUCIBILITY NOTES",
        "-" * 60,
        "1. Python 3.10+, dependencies in requirements.txt (pinned versions).",
        "2. CARE Farm A LOEO experiments (Finding 2) run on a single CPU in ~4 hours.",
        "3. Chronos-T5-tiny benchmark (Finding 3) runs CPU-only in ~3 hours.",
        "4. All random seeds fixed: numpy seed=42, XGBoost deterministic mode.",
        "5. Bootstrap uses np.random.default_rng(42); N_BOOT=2000 for significance,",
        "   N_BOOT=1000 for LAS-model CI.",
        "",
        f"Statement generated: {today}",
    ]
    return "\n".join(lines)


def generate_bibtex() -> str:
    entries = [ds["bibtex"] for ds in DATASETS]
    # Add Chronos model reference
    entries.append("""\
@article{chronos2024,
  author  = {Ansari, A. F. and others},
  title   = {Chronos: Learning the Language of Time Series},
  journal = {arXiv preprint arXiv:2403.07815},
  year    = {2024},
  url     = {https://arxiv.org/abs/2403.07815},
}""")
    # Add IEC 61400 standard reference
    entries.append("""\
@techreport{iec61400,
  title       = {{IEC} 61400-12-1: Wind energy generation systems --
                 Part 12-1: Power performance measurements of electricity
                 producing wind turbines},
  institution = {International Electrotechnical Commission},
  year        = {2017},
  number      = {IEC 61400-12-1:2017},
}""")
    return "\n\n".join(entries)


def generate_manifest() -> dict:
    manifest = {
        "project": "AetherGrid Wind Turbine Anomaly Detection",
        "generated": datetime.date.today().isoformat(),
        "target_venue": "IEEE Access",
        "datasets": [],
        "code_artifacts": [],
        "result_artifacts": [],
    }
    for ds in DATASETS:
        manifest["datasets"].append({
            "name": ds["name"],
            "doi": ds["doi"],
            "license": ds["license"],
        })
    for art in CODE_ARTIFACTS:
        p = Path(art["file"])
        manifest["code_artifacts"].append({
            "file": art["file"],
            "role": art["role"],
            "sha256_short": _sha256_short(p),
            "exists": p.exists(),
        })
    for rf in RESULT_ARTIFACTS:
        p = Path(rf)
        manifest["result_artifacts"].append({
            "file": rf,
            "sha256_short": _sha256_short(p),
            "exists": p.exists(),
        })
    return manifest


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data availability statement
    stmt = generate_data_availability_statement()
    stmt_path = OUT_DIR / "data_availability_statement.txt"
    stmt_path.write_text(stmt, encoding="utf-8")
    print(f"[OK] {stmt_path}")
    print("\n" + stmt[:800] + "\n  [... truncated — see full file]\n")

    # 2. BibTeX
    bib = generate_bibtex()
    bib_path = OUT_DIR / "dataset_references.bib"
    bib_path.write_text(bib, encoding="utf-8")
    print(f"[OK] {bib_path}  ({len(DATASETS) + 2} entries)")

    # 3. Manifest JSON
    manifest = generate_manifest()
    mf_path = OUT_DIR / "reproducibility_manifest.json"
    mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] {mf_path}")

    missing = [a for a in manifest["code_artifacts"] + manifest["result_artifacts"]
               if not a["exists"]]
    if missing:
        print(f"\n[WARN] {len(missing)} artifact(s) not yet on disk:")
        for m in missing:
            print(f"  - {m['file']}")
    else:
        print("\n[OK] All artifacts present.")


if __name__ == "__main__":
    main()
