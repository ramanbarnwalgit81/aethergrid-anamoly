# AetherGrid Anomaly Detection — Results Report

**Author:** Raman Barnwal
**Date:** Thursday, April 16, 2026
**Project:** AetherGrid — Per-turbine Isolation Forest anomaly detection on West Texas wind farm SCADA

## Abstract
(2–3 sentences. Problem, approach, headline result.)

## Dataset
- 12 turbines in a simulated 4×3 grid around Roscoe, TX (32.45°N, -100.50°W)
- 30 days at 5-min native resolution = 103,680 events total
- Meteorology source: NREL WIND Toolkit year 2012 / *or* deterministic synthetic West Texas profile
- Mechanical channels (gearbox temp, bearing temp, vibration): physics-synthesized from wind and load
- Fault injection: 4 windows in the final 7 days (gearbox spike ×2, vibration anomaly, bearing fault)

## Method
1. Per-turbine StandardScaler + IsolationForest(contamination=0.03, n_estimators=200)
2. Train on first 23 days (clean), hold out last 7 days for evaluation
3. Severity 0–100: percentile rank against training score distribution
4. SHAP TreeExplainer (fallback KernelExplainer) for top-3 contributing features per alert
5. Alert threshold: severity ≥ 75

## Findings (auto-generated from `findings.json`)

### Finding 1 — Detection rate
- Recall on injected rows: **XX.X%** (target ≥ 90%)
- FPR on clean rows: **X.X%** (target ≤ 5%)
- **PASS**

### Finding 2 — Alert fatigue reduction
- Global rule alerts (gearbox>85 OR vib>2.0): **N**
- Isolation Forest alerts: **N**
- Reduction: **XX.X%** (target > 60%)
- **PASS**

### Finding 3 — Scoring latency
- Mean: **X.X ms** per 12-turbine micro-batch
- p95: **X.X ms**
- Target: mean < 2000 ms → **PASS**

### Finding 4 — SHAP alignment
- True positives explained: **N**
- Top-2 SHAP includes expected sensor: **XX.X%** (target > 85%)
- **PASS**

### Finding 5 — Severity monotonicity
- Mean Spearman ρ across 4 fault windows: **0.XX** (target > 0.6)
- **PASS**

## Limitations
- Mechanical channels are physics-synthesized, not measured SCADA
- Single 30-day window; seasonal drift not evaluated
- Scoring is batch-offline, not true streaming (latency is compute-only)
- Only 3 fault types; real turbines have many more failure modes

## Reproduction
```bash
python -m src.data.fetch_nrel
python -m src.data.build_dataset
python -m src.model.train_if
python -m src.model.score_if
python -m src.model.shap_explain
python -m src.eval.run_all
streamlit run src/dashboard/app.py
```

## References
- IEC 61400-12-1 (power curve standard)
- NREL WIND Toolkit — Draxl et al. 2015
- Liu, Ting, Zhou — "Isolation Forest" ICDM 2008
- Lundberg & Lee — "A Unified Approach to Interpreting Model Predictions" NeurIPS 2017
