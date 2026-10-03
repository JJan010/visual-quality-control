# PatchCore localization evaluation

## Protocol

- Dataset: all 83 MVTec AD bottle test images.
- Evaluation resolution: 256 x 256.
- Ground-truth masks resized using nearest-neighbor interpolation.
- Normal images assigned all-zero masks.
- Scores: raw Anomalib anomaly maps.
- No per-image map normalization.
- Metrics computed jointly over all test pixels.
- No pixel-level decision threshold selected.
- Model and image-level threshold unchanged.

## Results

| Metric | Value |
|---|---:|
| Total pixels | 5439488 |
| Defect pixels | 314384 |
| Defect pixel fraction | approximately 5.78% |
| Pixel AUROC | 0.984735 |
| Pixel Average Precision | 0.759187 |

Full-precision values and checkpoint SHA-256 are stored in metrics.json.

## Visual inspection

The preview contains the lowest-scoring anomalous image from each defect
category and the highest-scoring normal image. All four maps share the
same color scale.

In the three selected anomalous images, strong responses overlap the
annotated defects. Responses extend beyond some defect boundaries.
The selected normal image has a weaker response.

These examples are diagnostic selections, not a representative sample.

## Interpretation

Image-level classification was perfect on this test set, while
pixel-level localization remains imperfect.

Pixel AUROC and Average Precision evaluate ranking across thresholds.
They are not pixel accuracy, Dice, or IoU at a fixed threshold.

Results apply to resized masks at 256 x 256. They do not establish
boundary accuracy at the original image resolution.

Pooled metrics give larger defects more influence through their greater
pixel count. The test set has already been inspected during development.
