# Statistical significance — event-level AUC

Bootstrap 95% CI (3000 resamples) and pairwise DeLong test (paired AUC comparison on the same datasets).

## Per-model AUC

| Model | n | AUC [95% CI] |
|---|---|---|
| PCARecon | 95 | 0.627 [0.501, 0.743] |
| Mahalanobis | 95 | 0.651 [0.532, 0.759] |
| Ensemble | 95 | 0.655 [0.537, 0.764] |
| GDN | 37 | 0.626 [0.437, 0.800] |

## Pairwise DeLong p-values (AUC difference)

| Model A | Model B | AUC_A | AUC_B | p | significant (p<0.05) |
|---|---|---|---|---|---|
| PCARecon | Mahalanobis | 0.627 | 0.651 | 0.209 | no |
| PCARecon | Ensemble | 0.627 | 0.655 | 0.152 | no |
| PCARecon | GDN | — | — | n/a (different farms) | — |
| Mahalanobis | Ensemble | 0.651 | 0.655 | 0.548 | no |
| Mahalanobis | GDN | — | — | n/a (different farms) | — |
| Ensemble | GDN | — | — | n/a (different farms) | — |