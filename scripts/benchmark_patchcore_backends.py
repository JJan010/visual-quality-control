import argparse
import csv
import itertools
import json
import platform
import random
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

from benchmark_patchcore_onnx import (
    infer_to_cpu,
    measure_ms,
    summarize,
)
from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.onnx_features import (
    OnnxCudaFeatureExtractor,
    sha256_file,
)
from visual_quality.inference.patchcore import PatchcorePredictor
from visual_quality.inference.tensorrt_features import (
    TensorRTCudaFeatureExtractor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    export_dir = args.export_dir.resolve()
    engine_dir = args.engine_dir.resolve()
    parity_path = args.parity_report.resolve()

    warmup = 30
    iterations = 120
    seed = 42

    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if (
        parity["status"] != "passed"
        or parity["torch_precision"] != "ieee"
        or parity["ort_use_tf32"] is not False
        or parity["tensorrt_precision"] != "fp32_tf32_off"
        or parity["blur_backend"] != "original_2d"
    ):
        raise RuntimeError("Parity report does not cover this configuration.")

    checks = {item["batch_size"]: item for item in parity["results"]}
    for batch_size in (1, 8):
        check = checks[batch_size]
        if (
            check["checked_images"] != 83
            or check["changed_decisions_vs_saved"] != 0
            or any(
                item["changed_decisions"] != 0
                for item in check["comparisons"].values()
            )
        ):
            raise RuntimeError("Missing successful parity check.")

    checkpoint_sha = sha256_file(run_dir / "model.pt")
    if checkpoint_sha != parity["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the parity check.")

    torch.cuda.set_device(0)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"

    predictor = PatchcorePredictor(
        run_dir, device="cuda:0", blur_backend="original_2d"
    )
    if predictor.threshold != parity["threshold"]:
        raise RuntimeError("Threshold differs from the parity check.")

    original_extractor = predictor.model.feature_extractor
    onnx_extractor = OnnxCudaFeatureExtractor(
        export_dir / "feature_extractor.onnx",
        expected_sha256=parity["onnx_sha256"],
        device="cuda:0",
    )
    trt_extractor = TensorRTCudaFeatureExtractor(engine_dir)

    if trt_extractor.engine_sha256 != parity["engine_sha256"]:
        raise RuntimeError("Engine differs from the parity check.")
    if trt_extractor.report["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("Engine report refers to another checkpoint.")
    if trt_extractor.report["onnx_sha256"] != onnx_extractor.onnx_sha256:
        raise RuntimeError("Engine report refers to another ONNX model.")

    extractors = {
        "pytorch": original_extractor,
        "onnx_iobinding": onnx_extractor,
        "tensorrt_fp32": trt_extractor,
    }

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        root / split["dataset_root"], predictor.image_size
    )
    samples = [dataset[index] for index in range(len(dataset))]
    images = [sample["image"] for sample in samples]

    # Every ordering occurs equally often.
    orders = list(itertools.permutations(extractors)) * (iterations // 6)
    random.Random(seed).shuffle(orders)

    rows = []
    results = {}

    print("GPU:", torch.cuda.get_device_name(0))
    print("Precision: IEEE FP32 / TensorRT TF32 disabled")
    print("Blur backend: original_2d")
    print("TensorRT profile:", trt_extractor.report["optimization_profile"])
    print("Warmup calls per variant:", warmup)
    print("Measured calls per variant:", iterations)

    try:
        for batch_size in (1, 8):
            batches = [
                torch.stack([
                    images[(start + offset) % len(images)]
                    for offset in range(batch_size)
                ])
                for start in range(0, len(images), batch_size)
            ]

            print(f"\nBatch {batch_size}: warmup...", flush=True)
            for index in range(warmup):
                batch = batches[index % len(batches)]
                for name in orders[index]:
                    predictor.model.feature_extractor = extractors[name]
                    infer_to_cpu(predictor, batch)
            torch.cuda.synchronize()

            timings = {name: [] for name in extractors}
            print(f"Batch {batch_size}: measuring...", flush=True)

            for round_index, order in enumerate(orders):
                batch_index = round_index % len(batches)
                batch = batches[batch_index]

                for position, name in enumerate(order):
                    predictor.model.feature_extractor = extractors[name]
                    latency_ms = measure_ms(predictor, batch)
                    timings[name].append(latency_ms)
                    rows.append({
                        "batch_size": batch_size,
                        "round": round_index,
                        "position": position,
                        "input_batch_index": batch_index,
                        "backend": name,
                        "latency_ms": latency_ms,
                    })

                if (round_index + 1) % 30 == 0:
                    print(
                        f"  Rounds: {round_index + 1}/{iterations}",
                        flush=True,
                    )

            variants = {
                name: summarize(values, batch_size)
                for name, values in timings.items()
            }
            reference_ms = variants["pytorch"]["mean_batch_ms"]

            for name, item in variants.items():
                item["speedup_vs_pytorch"] = (
                    reference_ms / item["mean_batch_ms"]
                )
                print(
                    f"{name}: "
                    f"mean={item['mean_batch_ms']:.3f} ms | "
                    f"p50={item['p50_batch_ms']:.3f} ms | "
                    f"p95={item['p95_batch_ms']:.3f} ms | "
                    f"images/s={item['serial_images_per_second']:.2f} | "
                    f"speedup={item['speedup_vs_pytorch']:.3f}x"
                )

            results[str(batch_size)] = variants
    finally:
        predictor.model.feature_extractor = original_extractor

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "benchmarks" / f"backends_fp32_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "checkpoint_sha256": checkpoint_sha,
        "onnx_sha256": onnx_extractor.onnx_sha256,
        "engine_sha256": trt_extractor.engine_sha256,
        "parity_report_sha256": sha256_file(parity_path),
        "benchmark_script_sha256": sha256_file(Path(__file__)),
        "timing_helpers_sha256": sha256_file(
            root / "scripts/benchmark_patchcore_onnx.py"
        ),
        "scope": (
            "Prepared CPU RGB batch -> H2D -> normalization -> "
            "complete PatchCore -> scores, labels and maps on CPU"
        ),
        "excluded": [
            "model loading", "disk I/O", "decoding", "resizing",
            "CPU batch creation", "backend selection",
        ],
        "input_paths": [sample["path"] for sample in samples],
        "batch_construction": "Sequential; final batch wraps to the start.",
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "torch", "torchvision", "anomalib",
                "onnxruntime-gpu", "tensorrt-cu12",
            )
        },
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_matmul_precision": torch.backends.cuda.matmul.fp32_precision,
        "torch_conv_precision": torch.backends.cudnn.conv.fp32_precision,
        "ort_use_tf32": False,
        "tensorrt_precision": "fp32_tf32_off",
        "tensorrt_execution": "dedicated stream, synchronized before return",
        "tensorrt_profile": trt_extractor.report["optimization_profile"],
        "blur_backend": "original_2d",
        "warmup_calls_per_variant": warmup,
        "measured_calls_per_variant": iterations,
        "order_seed": seed,
        "order_method": "All six permutations repeated equally, then shuffled.",
        "results": results,
    }

    (output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "latencies.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    shutil.copy2(parity_path, output_dir / "tensorrt_parity.json")
    shutil.copy2(
        engine_dir / "build_report.json",
        output_dir / "engine_build_report.json",
    )

    print("Output directory:", output_dir)
    print("PATCHCORE BACKEND BENCHMARK: OK")


if __name__ == "__main__":
    main()
