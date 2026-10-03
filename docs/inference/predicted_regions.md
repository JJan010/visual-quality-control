# Predicted region contours

The optional localization calibration converts a raw anomaly map to a binary
mask using `anomaly_map > localization_threshold`. It does not change the
image-level score, threshold or decision.

The default contour toggle is **Show predicted region**. Set overlay opacity
to zero to inspect the contour on the resized input without heatmap colors.
The turquoise line marks the boundary of the thresholded mask, including holes.
A dark graphic outline improves contrast; it does not alter the stored mask.
No connected components are removed and no mask smoothing is applied.

## Calibration

The threshold is the 95th percentile (`higher`) of the maximum map value from
each of the 42 normal validation images, evaluated with batch size 1.
No test labels or defect masks are used to select it.

For the current run:

- Localization threshold (rounded): 31.18199730.
- Calibration images with any flagged pixel: 2/42 (4.76%).
- Image classification threshold: 32.44375991821289, unchanged.
- Runtime: TensorRT FP32, IEEE PyTorch operations, separable blur.

Full threshold precision is read from the calibration file, never from the
rounded number printed in the console.

This targets image-wise occurrence of any flagged pixel on normal calibration
images. It is not a 5% pixel false-positive rate or a guarantee for unseen images.
The threshold is calibrated; localization quality still needs evaluation.

## Start with localization enabled

From the repository root:

    VQC_CONFIG="$PWD/configs/runtime/patchcore_tensorrt.json" \
    VQC_LOCALIZATION_CALIBRATION="$PWD/artifacts/runs/patchcore_20261003_135714_557111/localization_calibration/20261003_182936_190431/threshold.json" \
    python -m uvicorn visual_quality.api.app:app \
      --host 127.0.0.1 --port 8000 --workers 1

At startup the calibration runtime metadata, split hash, configuration hash,
resolution and decision rule must match. Incompatible calibration prevents
startup. Without the environment variable, classification and heatmaps still
work, but the UI states that region calibration is not loaded.

## API and display

With `include_visualization=true`, `visualization.localization` is either null
or contains the threshold, calibration SHA-256, mask pixel count, mask fraction,
a binary grayscale PNG and an RGBA contour PNG in model-input coordinates.

The displayed region area is a fraction of the resized image, not a physical
measurement in millimeters. No probability or segmentation-accuracy claim is
attached to it. The raw mask uses only pixel values 0 and 255.

Image classification and localization are separate rules. A classified anomaly
can have no map pixels above the localization threshold; a normal-classified
image can still contain flagged pixels. Both results are displayed honestly.

The contour is a model prediction, NOT the dataset ground-truth outline shown
in earlier offline evaluation figures. No ground truth is read by the API.

The heatmap retains per-image color normalization. The contour instead uses a
fixed threshold on raw values, so its mask does not depend on that color scale
or on the opacity slider. Report downloads omit PNG payloads but retain
localization metadata and the contour toggle state. Overlay downloads reflect
the current display controls. Binary masks can be downloaded separately.

## Check

With the server running:

    python scripts/check_api_localization.py

This compares image decisions/scores with and without visualization on two
known images, validates the calibration identity and threshold in the response,
and checks mask values, dimensions, pixel counts and contour transparency.
It does not measure segmentation quality against ground-truth masks.
