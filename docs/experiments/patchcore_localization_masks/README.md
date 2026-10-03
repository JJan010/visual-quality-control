# Fixed-threshold localization evaluation

## Protocol

- Dataset: MVTec AD bottle, all 83 test images.
- Defective images: 63. Normal images: 20.
- Runtime: TensorRT FP32, IEEE PyTorch operations, separable blur.
- Batch size: 1.
- Evaluation resolution: 256 x 256.
- Ground-truth resizing: Pillow NEAREST.
- Localization threshold: 31.181997299194336.
- Mask rule: anomaly_map > threshold.
- The threshold was calibrated on 42 normal validation images.
- No threshold tuning was performed on the test set.
- Predicted masks use the same implementation as the API and GUI.

## Results

| Metric | Value |
| --- | ---: |
| Pooled pixel IoU | 0.457132 |
| Pooled pixel Dice | 0.627441 |
| Pooled pixel precision | 0.462192 |
| Pooled pixel recall | 0.976615 |
| Mean per-image IoU, defective images only | 0.437440 |
| Mean per-image Dice, defective images only | 0.587816 |
| Normal images with any predicted region | 0/20 |
| Correct image-level classifications | 83/83 |

Pooled metrics combine pixel counts across all test images.
Macro IoU and Dice give equal weight to each of the 63 defective images.
Undefined metrics are stored as null; empty normal masks are not assigned
perfect IoU or Dice to increase the average.

| Defect category | Pooled IoU | Pooled precision | Pooled recall |
| --- | ---: | ---: | ---: |
| broken_large | 0.450739 | 0.452052 | 0.993596 |
| broken_small | 0.356575 | 0.357345 | 0.993987 |
| contamination | 0.526378 | 0.542114 | 0.947737 |

## Interpretation

The predicted regions cover most defect pixels but also include substantial
non-defect areas within defective images.

There are 664296 predicted positive pixels and 314384 ground-truth positive
pixels. The total predicted area is approximately 2.11 times the annotated area.

For contamination/018.png, all 375 defect pixels are covered, but the predicted
region contains 5929 pixels. Recall is 1.0 and IoU is approximately 0.06325.
This illustrates why high recall does not imply accurate boundaries.

All 63 defective images have some overlap between the predicted mask and
ground truth. No regions were predicted on the 20 normal test images.

The supported interpretation is anomaly detection with approximate region
localization. These masks are not validated for precise defect-area measurement.

The calibration procedure targets the occurrence of any flagged pixel on normal
images; it does not directly optimize segmentation overlap.

## Limitations

This is one dataset category and a small normal-image test sample.
The test set has been reused for engineering checks, so it is not a new holdout.
These observations do not establish production performance.

The threshold is retained unchanged after evaluation.

## Reproduce

From the repository root, with the project environment active:

    python scripts/evaluate_localization_masks.py

Local model artifacts, calibration and dataset are required.

## Evidence

- metrics.json: metrics, protocol and runtime identity.
- per_image.csv: individual results and dataset file hashes.
- calibration.json: exact calibration used for evaluation.
