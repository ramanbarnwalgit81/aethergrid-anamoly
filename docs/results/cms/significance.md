# Statistical significance — event-level AUC

Bootstrap 95% CI (3000 resamples) and pairwise DeLong test (paired AUC comparison on the same datasets).

## Per-model AUC

| Model | n | AUC [95% CI] |
|---|---|---|
| PCARecon | 95 | 0.627 [0.501, 0.743] |
| Mahalanobis | 95 | 0.651 [0.532, 0.759] |
| IsolationForest | 95 | 0.693 [0.580, 0.801] |
| OCSVM | 95 | 0.523 [0.400, 0.640] |
| KNN | 95 | 0.606 [0.481, 0.724] |
| LOF | 95 | 0.636 [0.511, 0.750] |
| ECOD | 95 | 0.615 [0.491, 0.734] |
| Ensemble | 95 | 0.655 [0.537, 0.764] |
| GDN | 37 | 0.626 [0.437, 0.800] |

## Pairwise DeLong p-values (AUC difference)

| Model A | Model B | AUC_A | AUC_B | p | significant (p<0.05) |
|---|---|---|---|---|---|
| PCARecon | Mahalanobis | 0.627 | 0.651 | 0.209 | no |
| PCARecon | IsolationForest | 0.627 | 0.693 | 0.232 | no |
| PCARecon | OCSVM | 0.627 | 0.523 | 0.333 | no |
| PCARecon | KNN | 0.627 | 0.606 | 0.105 | no |
| PCARecon | LOF | 0.627 | 0.636 | 0.502 | no |
| PCARecon | ECOD | 0.627 | 0.615 | 0.717 | no |
| PCARecon | Ensemble | 0.627 | 0.655 | 0.152 | no |
| PCARecon | GDN | — | — | n/a (different farms) | — |
| Mahalanobis | IsolationForest | 0.651 | 0.693 | 0.430 | no |
| Mahalanobis | OCSVM | 0.651 | 0.523 | 0.226 | no |
| Mahalanobis | KNN | 0.651 | 0.606 | 0.055 | no |
| Mahalanobis | LOF | 0.651 | 0.636 | 0.480 | no |
| Mahalanobis | ECOD | 0.651 | 0.615 | 0.341 | no |
| Mahalanobis | Ensemble | 0.651 | 0.655 | 0.548 | no |
| Mahalanobis | GDN | — | — | n/a (different farms) | — |
| IsolationForest | OCSVM | 0.693 | 0.523 | 0.075 | no |
| IsolationForest | KNN | 0.693 | 0.606 | 0.122 | no |
| IsolationForest | LOF | 0.693 | 0.636 | 0.313 | no |
| IsolationForest | ECOD | 0.693 | 0.615 | 0.119 | no |
| IsolationForest | Ensemble | 0.693 | 0.655 | 0.467 | no |
| IsolationForest | GDN | — | — | n/a (different farms) | — |
| OCSVM | KNN | 0.523 | 0.606 | 0.447 | no |
| OCSVM | LOF | 0.523 | 0.636 | 0.287 | no |
| OCSVM | ECOD | 0.523 | 0.615 | 0.410 | no |
| OCSVM | Ensemble | 0.523 | 0.655 | 0.212 | no |
| OCSVM | GDN | — | — | n/a (different farms) | — |
| KNN | LOF | 0.606 | 0.636 | 0.090 | no |
| KNN | ECOD | 0.606 | 0.615 | 0.770 | no |
| KNN | Ensemble | 0.606 | 0.655 | 0.053 | no |
| KNN | GDN | — | — | n/a (different farms) | — |
| LOF | ECOD | 0.636 | 0.615 | 0.563 | no |
| LOF | Ensemble | 0.636 | 0.655 | 0.387 | no |
| LOF | GDN | — | — | n/a (different farms) | — |
| ECOD | Ensemble | 0.615 | 0.655 | 0.309 | no |
| ECOD | GDN | — | — | n/a (different farms) | — |
| Ensemble | GDN | — | — | n/a (different farms) | — |

## CARE score — bootstrap 95% CI

| Model | n | CARE [95% CI] |
|---|---|---|
| ECOD | 95 | 0.583 [0.544, 0.618] |
| Ensemble | 95 | 0.665 [0.610, 0.717] |
| IsolationForest | 95 | 0.610 [0.571, 0.647] |
| KNN | 95 | 0.643 [0.588, 0.695] |
| LOF | 95 | 0.639 [0.581, 0.694] |
| Mahalanobis | 95 | 0.653 [0.599, 0.706] |
| OCSVM | 95 | 0.643 [0.586, 0.698] |
| PCARecon | 95 | 0.664 [0.609, 0.718] |

## Paired CARE test vs best model (Ensemble)

| Model | CARE | p (paired bootstrap) | sig (p<0.05) |
|---|---|---|---|
| ECOD | 0.583 | 0.000 | YES |
| IsolationForest | 0.610 | 0.007 | YES |
| KNN | 0.643 | 0.026 | YES |
| LOF | 0.639 | 0.039 | YES |
| Mahalanobis | 0.653 | 0.423 | no |
| OCSVM | 0.643 | 0.014 | YES |
| PCARecon | 0.664 | 0.755 | no |