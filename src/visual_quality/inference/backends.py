import json
from pathlib import Path

import torch


def precision_settings() -> dict:
    """Read the process-wide PyTorch precision settings."""
    return {
        "global": torch.backends.fp32_precision,
        "matmul": torch.backends.cuda.matmul.fp32_precision,
        "convolution": torch.backends.cudnn.conv.fp32_precision,
    }


def configure_ieee_fp32() -> None:
    """Call explicitly at application startup, before creating predictors."""
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"


def require_ieee_fp32() -> None:
    settings = precision_settings()
    if any(value != "ieee" for value in settings.values()):
        raise RuntimeError(
            "This predictor requires IEEE FP32. "
            "Call configure_ieee_fp32() at application startup. "
            f"Current settings: {settings}"
        )


def create_feature_backend(
    original_extractor,
    *,
    backend,
    device,
    checkpoint_sha256,
    model_config,
    export_dir=None,
    engine_dir=None,
):
    """Return the feature extractor and its artifact metadata."""
    if backend == "pytorch":
        return original_extractor, {
            "format": "pytorch_checkpoint",
            "checkpoint_sha256": checkpoint_sha256,
        }

    if backend == "onnx":
        export_dir = Path(export_dir).resolve()
        manifest = json.loads(
            (export_dir / "runtime_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if manifest.get("schema_version") != 1:
            raise RuntimeError("Unsupported ONNX manifest version.")
        if manifest["checkpoint_sha256"] != checkpoint_sha256:
            raise RuntimeError("ONNX belongs to another checkpoint.")
        if manifest["model_config"] != model_config:
            raise RuntimeError("ONNX and checkpoint configurations differ.")

        # New manifests bind runtime loading to the validated export.
        # Legacy manifests retain their existing identity checks.
        if "pipeline_validation_sha256" in manifest:
            from visual_quality.inference.export_validation import load_validated_export
            load_validated_export(export_dir=export_dir, checkpoint_sha256=checkpoint_sha256,
                                  model_config=model_config)

        # Optional runtime dependencies are imported only when selected.
        from visual_quality.inference.onnx_features import (
            OnnxCudaFeatureExtractor,
        )

        extractor = OnnxCudaFeatureExtractor(
            export_dir / "feature_extractor.onnx",
            expected_sha256=manifest["onnx_sha256"],
            device=str(device),
        )
        return extractor, {
            "format": "onnx",
            "directory": str(export_dir),
            "onnx_sha256": extractor.onnx_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }

    if backend == "tensorrt":
        engine_dir = Path(engine_dir).resolve()
        report = json.loads(
            (engine_dir / "build_report.json").read_text(encoding="utf-8")
        )
        if report["checkpoint_sha256"] != checkpoint_sha256:
            raise RuntimeError("TensorRT engine belongs to another checkpoint.")
        if report["precision"] != "fp32_tf32_off":
            raise RuntimeError("Only the FP32 TensorRT engine is supported.")

        from visual_quality.inference.tensorrt_features import (
            TensorRTCudaFeatureExtractor,
        )

        extractor = TensorRTCudaFeatureExtractor(engine_dir)
        return extractor, {
            "format": "tensorrt_plan",
            "directory": str(engine_dir),
            "engine_sha256": extractor.engine_sha256,
            "onnx_sha256": extractor.report["onnx_sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "tensorrt_version": extractor.report["tensorrt_version"],
            "optimization_profile": extractor.report["optimization_profile"],
        }

    raise ValueError(f"Unknown backend: {backend}")
