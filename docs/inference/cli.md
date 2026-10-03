# Single-image inference

The CLI loads a runtime configuration, analyzes one image and saves
a JSON result with an optional raw anomaly map.

## Requirements

Run commands from the repository root with the project virtual
environment activated.

The provided TensorRT configuration requires locally generated artifacts:

- PatchCore checkpoint: model.pt.
- Calibration report: calibration/threshold.json.
- TensorRT engine: feature_extractor.plan.
- TensorRT build metadata: build_report.json.

These artifacts are not stored in Git. The supplied configuration refers
to one local experiment; update its paths for a different experiment.
The TensorRT engine must be compatible with the target environment.

## Usage

Analyze an image and save its anomaly map:

    python scripts/predict_image.py \
      --config configs/runtime/patchcore_tensorrt.json \
      --image data/mvtec_ad/bottle/test/broken_large/011.png \
      --save-map

Analyze an image without saving the map:

    python scripts/predict_image.py \
      --config configs/runtime/patchcore_tensorrt.json \
      --image data/mvtec_ad/bottle/test/good/017.png

Display available arguments:

    python scripts/predict_image.py --help

## Runtime configuration

The current application configuration selects:

- Backend: TensorRT.
- Device: cuda:0.
- PyTorch precision policy: IEEE FP32.
- TensorRT engine: FP32 with TF32 disabled at build time.
- Gaussian blur: separable implementation.

Artifact paths inside the JSON configuration are resolved relative to
the configuration file's directory.

Command-line paths are resolved relative to the current working directory.

The runtime loader also supports PyTorch and ONNX through the shared
predictor. Backend-specific artifact fields are validated:

- pytorch: no export_dir or engine_dir.
- onnx: export_dir required.
- tensorrt: engine_dir required.

Model and calibration consistency checks remain in the predictor.

## Outputs

Each invocation creates a separate directory under artifacts/predictions.

result.json contains:

- Input image path and original dimensions.
- Raw image anomaly score.
- Calibrated threshold and decision rule.
- Boolean anomaly decision.
- Anomaly-map dimensions and optional filename.
- Runtime configuration and artifact metadata.

With --save-map, anomaly_map.npy contains a two-dimensional float32
array at the model input resolution of 256 x 256.

Map coordinates refer to the resized model input, not the original image.

The map is saved without display normalization. Its values are anomaly
scores, not probabilities or a binary segmentation mask.

The image decision is:

    is_anomaly = score > threshold

The image-level threshold is not automatically a pixel-level threshold.

## Manual smoke checks

Configuration: TensorRT FP32 with separable blur.

| Image | Score | Threshold | Decision |
| --- | ---: | ---: | --- |
| test/broken_large/011.png | 54.92676163 | 32.44375992 | ANOMALY |
| test/good/017.png | 31.41547203 | 32.44375992 | NORMAL |

Both invocations completed successfully and saved a JSON result and
a raw anomaly map.

These two checks exercise the application entry point. They do not
replace the full-dataset backend and optimized-pipeline parity checks.

## Process lifecycle

Each CLI invocation loads the model once, analyzes one image and exits.
Separate invocations load the model again.

The CLI does not report inference latency. Process startup and model
loading must not be confused with the dedicated inference benchmarks.
