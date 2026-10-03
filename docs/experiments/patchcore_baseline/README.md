# PatchCore baseline — MVTec AD bottle

## Purpose

Establish a feature-based reference model before developing and
benchmarking the inference system.

## Experimental setup

- Dataset: MVTec AD, bottle category.
- Split: configs/splits/bottle_v1.json.
- Memory bank construction: 167 normal training images.
- Threshold calibration: 42 held-out normal images.
- Test: 63 anomalous and 20 normal images.
- Backbone: pretrained Wide ResNet-50-2.
- Feature layers: layer2 and layer3.
- Input: RGB, resized to 256 x 256 without center cropping.
- Normalization: ImageNet mean and standard deviation.
- Extracted patch vectors: 171008, each with 1536 dimensions.
- Coreset sampling ratio: 0.01.
- Memory bank: 1710 vectors, approximately 10.02 MiB in float32.
- Image score: Anomalib PatchcoreModel.pred_score, num_neighbors=9.
- Precision: float32.
- GPU: NVIDIA GeForce RTX 5080.
- Seed: 42.

## Calibration

The threshold is the 95th percentile of normal validation scores,
using NumPy's "higher" method.

An image triggers an alarm when score > threshold.

- Threshold: approximately 32.44375992; full value in threshold.json.
- Calibration alarms: 2/42 (4.76%).
- No threshold adjustment was performed during test evaluation.
- The threshold metadata includes the checkpoint SHA-256.

## Test results

| Metric | Result |
|---|---:|
| TP | 63 |
| FN | 0 |
| FP | 0 |
| TN | 20 |
| Recall | 100% |
| Precision | 100% |
| False positive rate | 0% |
| F1 | 1.0 |
| Image AUROC | 1.0 |

Detected anomalies:
- broken_large: 20/20
- broken_small: 22/22
- contamination: 21/21

Normal images triggering an alarm: 0/20.

## Preparation timings

- Feature extraction: 3.37 seconds.
- Coreset selection: 5.24 seconds.

These are measurements from one run, not inference latency benchmarks.
Memory bank size excludes backbone weights and other runtime allocations.

## Interpretation and limitations

PatchCore outperformed the pixel-reconstruction autoencoder baselines
on this bottle test set.

The result covers image-level classification for one dataset category.
Pixel-level localization was subsequently evaluated: AUROC 0.984735 and Average Precision 0.759187 at 256 x 256. See localization/README.md for the protocol and limitations.

The test set was previously inspected during autoencoder experiments.
This is an exploratory benchmark, not an untouched final holdout.
Zero false alarms on 20 normal test images does not establish a zero
false-alarm rate in production.

## Next steps

1. Inspect anomaly maps against ground-truth masks.
2. Consolidate model loading and preprocessing into reusable code.
3. Benchmark inference latency, throughput, and GPU memory.
4. Evaluate deployment optimizations while checking prediction parity.

## References

- Roth et al., Towards Total Recall in Industrial Anomaly Detection:
  https://arxiv.org/abs/2106.08265
- Anomalib PatchCore documentation:
  https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html
