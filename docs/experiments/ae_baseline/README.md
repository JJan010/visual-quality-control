# Autoencoder baseline: test evaluation

## Protocol
- Run: ae_20261003_121002_827675
- Dataset: MVTec AD, bottle
- Training: 167 normal images
- Validation: 42 normal images
- Test: 20 normal and 63 anomalous images
- Checkpoint selected by validation reconstruction MSE: epoch 19
- Image score: mean squared reconstruction error
- Threshold: 0.002670697867870331
- Calibration: validation-score 95th percentile, NumPy method="higher"
- Alarm rule: score > threshold
- The threshold was fixed before test evaluation.

## Test results
| Metric | Result |
|---|---:|
| True positives | 36 |
| False negatives | 27 |
| False positives | 8 |
| True negatives | 12 |
| Recall | 57.14% |
| Precision | 81.82% |
| False positive rate | 40.00% |
| F1 | 67.29% |

## Detection by defect category
| Category | Detected | Total |
|---|---:|---:|
| broken_large | 12 | 20 |
| broken_small | 14 | 22 |
| contamination | 10 | 21 |

## Interpretation
The baseline misses 27 of 63 defective images and raises false alarms
on 8 of 20 normal test images.

Inspection of four normal validation reconstructions showed blurred
bottle details and residual errors around normal edges and reflections.
These observations suggest limitations of the reconstruction approach,
but do not establish the cause of individual test errors.

## Limitations
The same validation set was used for checkpoint selection and threshold
calibration. Its false-alarm rate is not an independent estimate.

The test set is small and covers one object category.
Pixel-level localization has not yet been evaluated.

Test results must not be used to tune the threshold and then reported
as an independent evaluation of that tuned threshold.

## Files
- threshold.json: calibration settings and decision threshold
- metrics.json: test metrics and category-level results
- test_scores.csv: per-image labels, scores and decisions
