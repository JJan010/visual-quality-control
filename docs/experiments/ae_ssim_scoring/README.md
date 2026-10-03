# SSIM scoring with a fixed MSE-trained autoencoder

## Hypothesis
Structural similarity may provide better anomaly scores than pixel-wise
MSE for the same autoencoder reconstructions.

## Protocol
- Fixed checkpoint: epoch 250 of ae_20261003_125745_500618.
- The autoencoder was trained with MSE. No retraining was performed.
- Score: spatial and channel mean of 1 - SSIM.
- RGB input in [0, 1], Gaussian window 11x11, sigma 1.5.
- Valid windows without padding.
- Threshold calibrated on 42 normal validation images.
- Quantile: 0.95, method="higher".
- Alarm rule: score > threshold.
- Calibration alarms: 2/42.

Full settings and checkpoint SHA-256 are recorded in settings.json.

## Results
| Metric | MSE scoring | SSIM scoring |
|---|---:|---:|
| Image AUROC | 0.764286 | 0.534921 |
| True positives | 35 | 20 |
| False negatives | 28 | 43 |
| False positives | 2 | 7 |
| True negatives | 18 | 13 |
| Recall | 55.56% | 31.75% |
| Precision | 94.59% | 74.07% |
| False positive rate | 10.00% | 35.00% |
| F1 | 70.00% | 44.44% |

## Conclusion
This SSIM scoring variant did not improve the baseline.
Retain the 250-epoch autoencoder with MSE scoring as the reference.

The result does not evaluate training with an SSIM loss and is not
a reproduction of the full method proposed by Bergmann et al.

## Limitations
Basic numerical checks passed, but they are not a comprehensive
validation against an independent SSIM implementation.
The inspected test set is an exploratory benchmark, not an independent
final evaluation. Only the bottle category was evaluated.

## Reference
Bergmann et al., Improving Unsupervised Defect Segmentation by Applying
Structural Similarity to Autoencoders.
https://arxiv.org/abs/1807.02011
