"""
cms — Condition Monitoring System (clean rebuild).

A correct, leak-free, reproducible wind-turbine anomaly-detection pipeline built
on the *official* CARE protocol (per-dataset Normal-Behavior Modelling on the
shipped train/prediction split) plus Kelmarsh/Penmanshiel.

This package is intentionally separate from the legacy `src/benchmark` code,
which (a) discarded all 255/955 anonymized features for CARE Farms B/C, (b)
pooled independent turbine-event series and split them 60/40 by timestamp, and
(c) labelled whole event windows as per-row positives. Those bugs pin AUC near
chance regardless of model. Nothing here fabricates results: every number is
produced by an executed run and written to docs/results/cms/.
"""

__version__ = "0.1.0"
