# Reproducibility — CARE re-evaluation (`src/cms`)

Every number in the paper is produced by a script below and written to
`docs/results/cms/`. No result is hand-entered.

## Environment
- Python 3.12.6, Windows 11 (10.0.26200)
- numpy 1.26.4 · scipy 1.13.1 · scikit-learn 1.8.0 · pandas 2.2.2 · torch 2.11.0+cpu
- xgboost 3.2.0 · laplace-torch 0.2.2.2 · chronos-forecasting 2.2.2
- **Hardware: CPU only (no GPU).** Deep/graph models therefore use bounded configs
  (documented in code); a GPU would permit larger configurations.
- All models use fixed seed 42.

## Data
- CARE benchmark (Gück/Roelofs et al. 2024), Zenodo record 15846963 (CARE v6, DOI 10.5281/zenodo.15846963), CC-BY-SA-4.0.
  Place the extracted archive under `data/real_scada/care/extracted/Wind Farm {A,B,C}/`.
- No data is redistributed in this repo; only derived window specs and JSON metrics.

## Pipeline (commands)
```bash
# 0. sanity: loader sees all 81/252/952 sensor features for A/B/C
python -m src.cms.data

# 1. corrected-protocol detector comparison (10 detectors, all 95 datasets)
python -m src.cms.benchmark --farms A,B,C \
  --models PCARecon,Mahalanobis,IsolationForest,OCSVM,KNN,LOF,ECOD,Ensemble \
  --smooth none,0.05 --quantiles 0.90,0.95,0.97,0.99,0.995 --tag baselines
python -m src.cms.benchmark --farms A,B --models GDN --tag GDN_AB
python -m src.cms.benchmark --farms A,B,C --models TemporalNBM --tag ABC

# 2. statistical significance (DeLong AUC + paired-bootstrap CARE)
python -m src.cms.stats --tags baselines,GDN_AB

# 3. central finding: label sensitivity (status vs event vs precursor)
python -m src.cms.label_sensitivity --farms A,B,C --model Mahalanobis

# 4. CARE-Precursor predictive benchmark (released window spec)
python -m src.cms.precursor --farms A,B,C --model Mahalanobis --agg mean

# 5. early-warning lead time + per-fault-type detectability
python -m src.cms.leadtime --farms A,B,C --model Ensemble --quantile 0.99

# 6. merged final report (CIs across all detectors)
python -m src.cms.report --tags baselines,GDN_AB,ABC
```

## Outputs
- `docs/results/cms/benchmark_*.json` — per-config rows, calibrated points,
  per-dataset level scores + CARE contributions (for significance).
- `docs/results/cms/significance.md` — DeLong + bootstrap CARE tables.
- `docs/results/cms/label_sensitivity_*.json` — the label gap.
- `docs/results/cms/precursor_*.json` + `data/benchmark/care_precursor_v2/` —
  predictive task + released precursor windows.
- `docs/results/cms/leadtime_*.json/.png` — lead time / per-fault.
- `docs/results/cms/REPORT_final.md` — consolidated comparison.
