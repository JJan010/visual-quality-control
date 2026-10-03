# Autoencoder: longer training

## Hypothesis
Longer training may improve reconstruction of normal bottle details.

A diagnostic run on four fixed training images showed clearer bottle
rings after 5000 updates than after 1000 updates.

## Experiment
- Run: ae_20261003_125745_500618
- Increased training budget from 20 to 250 epochs.
- Kept architecture, MSE loss, learning rate, batch size and data split.
- Selected the checkpoint using normal-validation MSE.
- Best epoch: 250.
- Best validation MSE: 0.000508704084149074.
- Recalibrated the threshold on the same 42 normal validation images.
- Threshold rule: 95th percentile, method="higher".
- Image anomaly score: mean squared reconstruction error.

## Comparison
| Metric | 20 epochs | 250 epochs |
|---|---:|---:|
| Image AUROC | 0.629365 | 0.764286 |
| True positives | 36 | 35 |
| False negatives | 27 | 28 |
| False positives | 8 | 2 |
| True negatives | 12 | 18 |
| Recall | 57.14% | 55.56% |
| Precision | 81.82% | 94.59% |
| False positive rate | 40.00% | 10.00% |
| F1 | 67.29% | 70.00% |

AUROC was calculated from each run's test_scores.csv by comparing all
anomalous-normal score pairs, counting ties as half a correct ordering.

## Findings
Longer training improved validation reconstruction and test-score ranking.
It removed six false alarms without introducing new false alarms.
Five previously missed defects were detected, but six previously detected
defects were missed. Twenty-two defects were missed by both models.

## Limitations
The validation set serves both checkpoint selection and calibration.
The test set has already been inspected during development, so this is
an exploratory benchmark comparison, not an independent final evaluation.

## Next experiment
Keep the 250-epoch checkpoint fixed and investigate SSIM-based scoring.
