# CARE benchmark — final results (clean cms pipeline)

Merged tags: baselines, GDN_AB, ABC | farms: A,B,C

Published reference (official aggregate CARE score): **Random = 0.5**, **Isolation Forest (paper) = 0.45**, **Autoencoder (paper, SOTA) = 0.66**

## Best operating point per model (aggregate over all farms)

| Model | Best CARE | event-AUC [95% CI] | smooth | q | tp/fp/fn | per-farm CARE |
|---|---|---|---|---|---|---|
| Ensemble | **0.6646** ✅ | 0.655 [0.534, 0.760] | None | 0.95 | 35/17/9 | A:0.6638 B:0.6221 C:0.6862 |
| PCARecon | **0.6637** ✅ | 0.627 [0.500, 0.742] | None | 0.95 | 34/16/10 | A:0.6603 B:0.6216 C:0.6858 |
| Mahalanobis | **0.6533** | 0.651 [0.530, 0.756] | None | 0.95 | 34/16/10 | A:0.6747 B:0.6192 C:0.6636 |
| KNN | **0.6433** | 0.606 [0.482, 0.724] | None | 0.97 | 32/17/12 | A:0.6192 B:0.6177 C:0.6678 |
| OCSVM | **0.6432** | 0.523 [0.400, 0.640] | None | 0.95 | 33/23/11 | A:0.6461 B:0.5303 C:0.6761 |
| LOF | **0.6389** | 0.636 [0.510, 0.750] | None | 0.97 | 31/18/13 | A:0.5545 B:0.6016 C:0.677 |
| TemporalNBM | **0.6364** | 0.680 | None | 0.995 | 39/33/5 | A:0.6549 B:0.5941 C:0.6537 |
| IsolationForest | **0.6101** | 0.693 [0.581, 0.796] | None | 0.9 | 36/29/8 | A:0.6666 B:0.5216 C:0.6137 |
| GDN | **0.5991** | 0.797 [0.622, 0.938] | 0.05 | 0.995 | 9/0/8 | A:0.6072 B:0.583 |
| ECOD | **0.5833** | 0.615 [0.492, 0.732] | None | 0.9 | 32/27/12 | A:0.5476 B:0.5216 C:0.6109 |

## Calibrated operating point (most sensitive threshold with event-FP<=1)

| Model | CARE | event-AUC | smooth | q | tp/fp/fn |
|---|---|---|---|---|---|
| Ensemble | 0.6574 | 0.655 | None | 0.99 | 27/7/17 |
| PCARecon | 0.6564 | 0.627 | None | 0.99 | 27/7/17 |
| Mahalanobis | 0.6343 | 0.651 | None | 0.995 | 23/6/21 |
| KNN | 0.6316 | 0.606 | None | 0.995 | 22/7/22 |
| OCSVM | 0.6417 | 0.523 | None | 0.995 | 24/6/20 |
| LOF | 0.6364 | 0.636 | None | 0.995 | 25/7/19 |
| IsolationForest | 0.5743 | 0.693 | None | 0.995 | 15/1/29 |
| ECOD | 0.5639 | 0.615 | None | 0.995 | 17/5/27 |
| GDN | 0.5924 | 0.626 | None | 0.95 | 7/1/10 |
| TemporalNBM | 0.6364 | 0.680 | None | 0.995 | 39/33/5 |