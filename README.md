# The Reporting Gap in Wind-Turbine Condition Monitoring

Code and artifacts for the paper "The Reporting Gap in Wind-Turbine Condition
Monitoring: A Physics-Informed, Conformally-Calibrated Evaluation Framework"
(R. Barnwal). The paper examines how the choice of ground-truth label, rather
than model capacity, governs reported anomaly-detection performance on the CARE
wind-turbine benchmark, and reports honest numbers under each label convention
across 89 turbine-years from three wind farms.

This repository holds the corrected evaluation harness, the CARE-Precursor
labels, the physics-residual stacker, the conformal calibration layer, and every
result file behind the tables and figures. No reported number is hand-entered.
Each is written to a JSON file by the script that produces it.

## What is included

- `src/` source code.
- `docs/results/` all result files behind the paper (101 JSON files).
- `data/benchmark/care_precursor*/` the CARE-Precursor window definitions.
- `models/` pretrained per-turbine detector weights.
- `RELEASE_MANIFEST_SHA256.txt` SHA-256 and byte size of every result file.

## What is not included

The raw SCADA datasets. CARE, Kelmarsh, and Penmanshiel are third-party datasets
under their own licences and are not redistributed here. Only derived window
specifications and JSON metrics are stored in this repository. See "Data" below
for where to obtain them.

## Environment

Python 3.12.6, CPU only. No GPU was used. The deep and graph models run in
bounded configurations documented in the code, and a GPU would permit larger
ones. Core package versions: numpy 1.26.4, scipy 1.13.1, scikit-learn 1.8.0,
pandas 2.2.2, torch 2.11.0+cpu. All runs use fixed seed 42.

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS or Linux
pip install -r requirements.txt
```

## Data

Download the three datasets from Zenodo and place them as noted. Nothing here
re-hosts them.

- CARE (Gück, Roelofs, Faulstich, Data 9(12):138, 2024). Extract to
  `data/real_scada/care/extracted/Wind Farm {A,B,C}/`.
- Kelmarsh (Cubico Sustainable Investments, 2022), DOI 10.5281/zenodo.5841834.
- Penmanshiel (Cubico Sustainable Investments, 2022), DOI 10.5281/zenodo.5946808.

## Reproducing the paper

Full commands, flags, and run order are in `docs/paper/REPRODUCE.md`. The table
below maps each result to the script that produces it. Outputs are written under
`docs/results/`.

| Paper item | Script (run as `python -m ...`) | Output |
| --- | --- | --- |
| Table 3, Nair replication | `src.benchmark.care_sota` | `docs/results/care_sota.json` |
| Table 6 and Fig. 10, ten-detector plateau | `src.cms.benchmark` | `docs/results/cms/benchmark_ABC.json` |
| Fig. 9, GDN dependency graph | `src.cms.benchmark` (GDN) | `docs/results/cms/benchmark_GDN_AB.json` |
| Fig. 11, same-scores label gap | `src.cms.label_sensitivity` | `docs/results/cms/` |
| Table 7, Label Ambiguity Score | `src.benchmark.label_ambiguity_test` | `docs/results/label_ambiguity_test.json` |
| Fig. 6 and 7, PINN-v2 stacker | `src.benchmark.pinn_stacker` | `docs/results/care_pinn_results.json` |
| Table 5, conformal risk control | `src.cms.evaluate` | `docs/results/calibration.json` |
| CARE-Precursor labels | `src.benchmark.care_precursor` | `data/benchmark/care_precursor*/` |
| Fig. 14, labeling decision tree | `src.benchmark.labeling_decision_tree` | figure |
| Paper figures and table-charts | `src.cms.paper_figures`, `src.cms.paper_table_figures` | `docs/paper/figures/` |

## Verifying the artifacts

`RELEASE_MANIFEST_SHA256.txt` lists the SHA-256 hash and byte size of every file
under `docs/results/` and `data/benchmark/care_precursor*/`. To recompute the
hashes and compare:

```
python - <<'PY'
import hashlib, pathlib
for line in open("RELEASE_MANIFEST_SHA256.txt"):
    if line.startswith("#"): continue
    want, _size, rel = line.split(None, 2)
    rel = rel.strip()
    got = hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()
    print("OK " if got == want else "MISMATCH", rel)
PY
```

## Repository layout

```
src/cms/         corrected CARE re-evaluation harness: loader, CARE score,
                 ten detectors, statistics, figures
src/benchmark/   Nair replication, CARE-Precursor, PINN-v2 stacker,
                 Label Ambiguity Test, labeling decision tree
docs/results/    result JSON files behind every number in the paper
data/benchmark/  CARE-Precursor window definitions
models/          pretrained per-turbine detector weights
docs/paper/      manuscript, figures, and REPRODUCE.md
```

## Citation

```
@article{barnwal2026reportinggap,
  author  = {Barnwal, Raman},
  title   = {The Reporting Gap in Wind-Turbine Condition Monitoring:
             A Physics-Informed, Conformally-Calibrated Evaluation Framework},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

Code is released under the MIT License. Result files and the CARE-Precursor
labels are released under CC-BY-4.0. The underlying CARE, Kelmarsh, and
Penmanshiel SCADA datasets remain under their original licences.
