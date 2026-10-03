# Dataset: MVTec AD / bottle

## Source

Dataset:
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Reference:
Bergmann et al., "MVTec AD - A Comprehensive Real-World
Dataset for Unsupervised Anomaly Detection", CVPR 2019.

Dataset license: CC BY-NC-SA 4.0.
Dataset files are stored locally and excluded from Git.

## Verified file counts

| Directory | PNG files |
| --- | ---: |
| train/good | 209 |
| test/good | 20 |
| test/broken_large | 20 |
| test/broken_small | 22 |
| test/contamination | 21 |
| ground_truth/broken_large | 20 |
| ground_truth/broken_small | 22 |
| ground_truth/contamination | 21 |

## Initial inspection

Inspected sample: test/broken_large/000.png
Matching mask: ground_truth/broken_large/000_mask.png

- Image shape: (900, 900, 3), RGB, uint8.
- Mask shape: (900, 900), values: 0 and 255.
- Annotated defect: 88192 pixels.
- The overlay visually aligns with the damaged region.
- This preview displays dataset annotations, not model predictions.

## Evaluation protocol

We will reserve part of train/good for validation.
Training and validation filenames will be recorded explicitly.

Test images and masks will not be used to fit the model
or select the alarm threshold.

## Generate the preview

From the repository root, with the environment activated:

```bash
python scripts/preview_sample.py
```

Output: artifacts/previews/bottle_broken_large_000.png
