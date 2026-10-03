import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from PIL import Image

from visual_quality.inference.feature_export import PatchcoreFeatureExport
from visual_quality.inference.patchcore import PatchcorePredictor


ATOL = 1e-4
RTOL = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("STOP: PyTorch nie ma dostępu do GPU.")

    predictor = PatchcorePredictor(run_dir, device="cpu")

    if predictor.config["layers"] != ["layer2", "layer3"]:
        raise RuntimeError("STOP: nieoczekiwane warstwy ekstraktora.")
    if predictor.image_size != 256:
        raise RuntimeError("STOP: ten eksport zakłada obrazy 256 x 256.")

    model = PatchcoreFeatureExport(predictor.model.feature_extractor)
    model.eval()
    model.requires_grad_(False)

    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    dataset_root = project_root / split["dataset_root"]
    sample_paths = split["validation"][:8]

    if len(sample_paths) != 8:
        raise RuntimeError("STOP: potrzeba ośmiu obrazów walidacyjnych.")

    images = []
    for relative_path in sample_paths:
        with Image.open(dataset_root / relative_path) as image:
            images.append(predictor.prepare_image(image))

    normalized = predictor.normalize(torch.stack(images)).contiguous()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "exports" / f"features_onnx_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    onnx_path = output_dir / "feature_extractor.onnx"

    print("Export device: cpu")
    print("Checkpoint:", run_dir / "model.pt")
    print("Exporting feature extractor...")

    # Batch 2 is the example; the exported batch dimension is dynamic.
    torch.onnx.export(
        model,
        (normalized[:2],),
        str(onnx_path),
        input_names=["images"],
        output_names=["layer2", "layer3"],
        opset_version=18,
        dynamo=True,
        dynamic_shapes={
            "images": {0: torch.export.Dim("batch", min=1, max=8)}
        },
        external_data=False,
        report=True,
        artifacts_dir=str(output_dir),
    )

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)

    print("ONNX IR version:", graph.ir_version)
    print("ONNX opsets:", [(item.domain, item.version)
                           for item in graph.opset_import])

    ort.preload_dlls(cuda=True, cudnn=True, msvc=False)

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(output_dir / "ort_profile")

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})
        ],
    )
    session.disable_fallback()

    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("STOP: sesja nie korzysta z CUDA.")

    print("Session providers:", session.get_providers())
    comparisons = []

    try:
        for batch_size in (1, 8):
            batch = normalized[:batch_size]

            with torch.inference_mode():
                reference = [
                    tensor.cpu().numpy()
                    for tensor in model(batch)
                ]

            actual = session.run(
                ["layer2", "layer3"],
                {"images": batch.numpy()},
            )

            expected_shapes = [
                (batch_size, 512, 32, 32),
                (batch_size, 1024, 16, 16),
            ]

            for name, expected_shape, ref, result in zip(
                ("layer2", "layer3"),
                expected_shapes,
                reference,
                actual,
            ):
                if ref.shape != expected_shape or result.shape != expected_shape:
                    raise RuntimeError(f"STOP: niepoprawny kształt {name}.")
                if ref.dtype != np.float32 or result.dtype != np.float32:
                    raise RuntimeError(f"STOP: niepoprawny typ {name}.")
                if not np.isfinite(ref).all() or not np.isfinite(result).all():
                    raise RuntimeError(f"STOP: NaN lub Inf w {name}.")

                difference = np.abs(
                    result.astype(np.float64) - ref.astype(np.float64)
                )
                max_difference = float(difference.max())
                mean_difference = float(difference.mean())

                np.testing.assert_allclose(
                    result,
                    ref,
                    rtol=RTOL,
                    atol=ATOL,
                    err_msg=f"Batch {batch_size}, output {name}",
                )

                comparisons.append({
                    "batch_size": batch_size,
                    "output": name,
                    "shape": list(result.shape),
                    "max_absolute_difference": max_difference,
                    "mean_absolute_difference": mean_difference,
                })

                print(
                    f"Batch {batch_size} | {name} | "
                    f"shape={result.shape} | "
                    f"max diff={max_difference:.10f} | "
                    f"mean diff={mean_difference:.10f}"
                )
    finally:
        profile_path = Path(session.end_profiling())

    events = json.loads(profile_path.read_text(encoding="utf-8"))
    conv_providers = {
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node"
        and "Conv" in event.get("args", {}).get("op_name", "")
        and event.get("args", {}).get("provider")
    }

    if conv_providers != {"CUDAExecutionProvider"}:
        raise RuntimeError(
            f"STOP: niepotwierdzone wykonanie konwolucji na GPU: "
            f"{conv_providers}"
        )

    report = {
        "status": "passed",
        "checkpoint_sha256": sha256_file(run_dir / "model.pt"),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_ir_version": graph.ir_version,
        "versions": {
            name: version(name)
            for name in ("torch", "torchvision", "anomalib",
                         "onnx", "onnxscript", "onnxruntime-gpu")
        },
        "input": {
            "name": "images",
            "dtype": "float32",
            "shape": ["batch", 3, 256, 256],
            "intended_batch_range": [1, 8],
            "normalization_in_graph": False,
            "normalization_mean": predictor.config["normalization_mean"],
            "normalization_std": predictor.config["normalization_std"],
        },
        "sample_paths": sample_paths,
        "reference_device": "cpu",
        "runtime_device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "ort_use_tf32": False,
        "atol": ATOL,
        "rtol": RTOL,
        "conv_execution_providers": sorted(conv_providers),
        "comparisons": comparisons,
        "profile": profile_path.name,
    }

    (output_dir / "export_check.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Conv execution providers:", sorted(conv_providers))
    print("Output directory:", output_dir)
    print("PATCHCORE FEATURE EXPORT: OK")


if __name__ == "__main__":
    main()
