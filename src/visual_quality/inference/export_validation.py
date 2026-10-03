"""Validate the identity and scope of a per-export ONNX approval report.

No GPU imports: this gate can be tested independently of the inference stack.
Hashes detect accidental artifact replacement; reports are not cryptographically
signed certificates and must come from a trusted validation run.
"""
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_hashes(run_dir, export_dir):
    run_dir, export_dir = Path(run_dir), Path(export_dir)
    files = {
        "checkpoint": run_dir / "model.pt",
        "calibration": run_dir / "calibration/threshold.json",
        "split": run_dir / "split.json",
        "baseline_metrics": run_dir / "evaluation/metrics.json",
        "baseline_scores": run_dir / "evaluation/test_scores.csv",
        "onnx": export_dir / "feature_extractor.onnx",
        "export_check": export_dir / "export_check.json",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_validated_export(run_dir=None, export_dir=None, *,
                          checkpoint_sha256=None, model_config=None):
    """Check full evidence at build time, or portable model identity at runtime."""
    export_dir = Path(export_dir)
    report_path = export_dir / "pipeline_validation.json"
    manifest = json.loads((export_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1, "Unsupported runtime manifest.")
    require(manifest.get("pipeline_validation_sha256") == sha256_file(report_path),
            "ONNX validation report does not match the runtime manifest.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1, "kind": "patchcore_onnx_pipeline", "status": "passed",
        "torch_precision": "ieee", "ort_use_tf32": False,
        "blur_backend": "original_2d", "atol": 1e-4, "rtol": 1e-4,
    }
    for key, value in expected.items():
        require(report.get(key) == value, f"Unsupported ONNX validation: {key}.")
    require(math.isfinite(report["threshold"]), "Invalid validation threshold.")
    actual_onnx = sha256_file(export_dir / "feature_extractor.onnx")
    require(report["onnx_sha256"] == manifest["onnx_sha256"] == actual_onnx,
            "ONNX changed after validation.")
    require(report["checkpoint_sha256"] == manifest["checkpoint_sha256"],
            "Checkpoint identity differs between reports.")
    require(report["model_config"] == manifest["model_config"], "Model configuration differs.")
    if run_dir is not None:
        run_dir = Path(run_dir)
        require(report["evidence_hashes"] == evidence_hashes(run_dir, export_dir),
                "Validation inputs changed; validate a new export.")
        checkpoint_sha256 = sha256_file(run_dir / "model.pt")
        calibration = json.loads((run_dir / "calibration/threshold.json").read_text(encoding="utf-8"))
        model_config = calibration["config"]
        require(calibration["checkpoint_sha256"] == checkpoint_sha256,
                "Calibration belongs to another checkpoint.")
        require(report["threshold"] == calibration["threshold"], "Calibration threshold changed.")
    require(checkpoint_sha256 is not None and model_config is not None,
            "Expected checkpoint identity and model configuration.")
    require(report["checkpoint_sha256"] == checkpoint_sha256,
            "ONNX approval belongs to another checkpoint.")
    require(report["model_config"] == model_config, "ONNX approval uses another configuration.")

    pipeline = report["pipeline_checks"]
    require(len(pipeline) == 2 and {r["batch_size"] for r in pipeline} == {1, 8},
            "Incomplete pipeline batch coverage.")
    for result in pipeline:
        require(result["checked_images"] == 83, "Incomplete image coverage.")
        require(result.get("score_parity_passed") is True
                and result.get("map_parity_passed") is True, "Missing numerical parity checks.")
        require(result["changed_decisions"] == result["changed_decisions_vs_saved"] == 0,
                "Decision parity failed.")
        for name in ("max_score_difference", "max_map_difference"):
            value = result[name]
            require(math.isfinite(value) and value >= 0, "Invalid numerical diagnostic.")
    binding = report["binding_checks"]
    expected_cases = {(b, layer) for b in (1, 3, 8) for layer in ("layer2", "layer3")}
    require(len(binding) == 6 and {(r["batch_size"], r["layer"]) for r in binding} == expected_cases,
            "Incomplete I/O binding coverage.")
    for result in binding:
        require(result.get("parity_passed") is True, "I/O binding parity was not confirmed.")
        require(math.isfinite(result["max_difference"]) and result["max_difference"] >= 0,
                "Invalid binding diagnostic.")
    return report
