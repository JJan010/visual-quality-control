# GUI and localization verification

Runtime: TensorRT FP32 with separable Gaussian blur.

## Calibration

The localization threshold was selected using the 95th percentile
(method: higher) of maximum anomaly-map values from 42 normal
validation images.

- Localization threshold, rounded: 31.18199730.
- Calibration images with any flagged pixel: 2/42 (4.76%).
- Image classification threshold: 32.44375991821289, unchanged.

This is an observed image-level alarm rate on the calibration set,
not a pixel false-positive rate or a guarantee on new data.

## API verification

| Image | Image decision | Flagged pixels | Score parity |
| --- | --- | ---: | --- |
| test/broken_large/011.png | Anomaly | 8275 | Passed |
| test/good/017.png | Normal | 0 | Passed |

Masks use the 256 x 256 model-input coordinate system.

The check verified calibration identity, score agreement with and
without visualization, unchanged image decisions, binary mask values,
dimensions, pixel counts and contour transparency.

## Manual visual inspection

The defective bottle was inspected with both zero and nonzero
heatmap opacity. The predicted contour remained visible and outlined
the lower damaged region.

This visual check is not a quantitative segmentation evaluation.
The displayed contour is derived from the model map, not ground truth.

## Evidence

- localization_threshold.json: calibration and runtime metadata.
- validation_map_maxima.csv: per-image calibration measurements.
- api_localization_report.json: API verification results.

See ../gui.md for the interface and ../predicted_regions.md for
localization behavior and startup instructions.
