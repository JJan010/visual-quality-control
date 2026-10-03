# Unified PatchCore inference

## Public interface

PatchcorePredictor selects the feature extractor through its backend
argument. All backends return the same PatchcorePrediction structure:

- scores: image-level anomaly scores;
- labels: boolean decisions using score > threshold;
- anomaly_maps: spatial anomaly scores.

Outputs remain on the inference device.

Supported backends:

| Backend | Required artifacts |
|---|---|
| pytorch | run_dir with checkpoint and calibration |
| onnx | run_dir plus export_dir |
| tensorrt | run_dir plus engine_dir |

The default remains backend="pytorch". Existing calls continue to work.

## Precision policy

Call configure_ieee_fp32() explicitly at application startup before
creating predictors that require IEEE FP32.

    from visual_quality.inference.backends import configure_ieee_fp32
    from visual_quality.inference.patchcore import PatchcorePredictor

    configure_ieee_fp32()

    predictor = PatchcorePredictor(
        run_dir,
        device="cuda:0",
        backend="tensorrt",
        engine_dir=engine_dir,
        precision="ieee",
        blur_backend="original_2d",
    )

PyTorch precision settings are process-wide. Predictor construction
does not silently change them.

The legacy PyTorch default uses precision="inherit".
ONNX and TensorRT require precision="ieee" and cuda:0.
Required IEEE settings are checked at construction and prediction.

## Artifact validation

The predictor verifies the checkpoint against calibration metadata
and checks the recorded model dependency versions.

ONNX additionally requires runtime_manifest.json beside
feature_extractor.onnx. The manifest links the ONNX hash, checkpoint
hash and model configuration.

TensorRT requires feature_extractor.plan and build_report.json.
The adapter checks the engine hash, build version, GPU and I/O contract.
The predictor checks the engine's checkpoint association.

The archived onnx_runtime_manifest.json belongs to the exact export
identified by its hashes. A new export requires corresponding metadata.

## Runtime metadata

predictor.runtime_metadata() returns the selected backend, device,
precision policy, current precision settings, blur implementation,
threshold and artifact identifiers.

The feature backend is selected before moving the complete model to
the inference device, avoiding an unused PyTorch GPU backbone when
ONNX or TensorRT is selected.

The adapters currently support sequential inference. ONNX is bound
to the CUDA stream selected at initialization. TensorRT uses a
dedicated stream and synchronizes before returning its outputs.

## Verification

Legacy PyTorch check:

- 83 images;
- exact preprocessing match;
- maximum score difference: 0;
- changed decisions: 0.

Unified interface check:

- 83 images separately at batch sizes 1 and 8;
- PyTorch IEEE FP32 as reference;
- ONNX and TensorRT scores and maps passed atol=1e-4, rtol=1e-4;
- changed decisions: 0.

See the accompanying JSON reports for exact measurements.

This integration check covers original_2d blur. Combined deployment
of accelerated feature backends with separable_1d remains to be checked.
FP16 is outside the current scope.
