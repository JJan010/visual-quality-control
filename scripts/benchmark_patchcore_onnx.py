import argparse
import csv
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.onnx_features import (
    OnnxCudaFeatureExtractor,
    sha256_file,
)
from visual_quality.inference.patchcore import PatchcorePredictor


@torch.inference_mode()
def infer_to_cpu(predictor, images):
    prediction = predictor.predict_batch(images)
    return (
        prediction.scores.cpu(),
        prediction.labels.cpu(),
        prediction.anomaly_maps.cpu(),
    )


def measure_ms(predictor, images):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()

    outputs = infer_to_cpu(predictor, images)

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    del outputs
    return elapsed_ms


def summarize(values, batch_size):
    values = np.asarray(values, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "calls": len(values),
        "mean_batch_ms": mean_ms,
        "p50_batch_ms": float(np.percentile(values, 50)),
        "p95_batch_ms": float(np.percentile(values, 95)),
        "serial_images_per_second": batch_size * 1000 / mean_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 2:
        raise SystemExit("Use warmup >= 1 and iterations >= 2.")

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    export_dir = args.export_dir.resolve()
    parity_path = args.parity_report.resolve()

    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if (
        parity["status"] != "passed"
        or parity["torch_precision"] != "ieee"
        or parity["ort_use_tf32"] is not False
        or parity["blur_backend"] != "original_2d"
    ):
        raise RuntimeError("Parity report does not cover this configuration.")

    checks = {
        item["batch_size"]: item for item in parity["pipeline_checks"]
    }
    for batch_size in (1, 8):
        check = checks[batch_size]
        if (
            check["checked_images"] != 83
            or check["changed_decisions"] != 0
            or check["changed_decisions_vs_saved"] != 0
        ):
            raise RuntimeError("Missing successful pipeline parity check.")

    checkpoint_sha = sha256_file(run_dir / "model.pt")
    if checkpoint_sha != parity["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the parity check.")

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
    adapter = OnnxCudaFeatureExtractor(
        export_dir / "feature_extractor.onnx",
        expected_sha256=parity["onnx_sha256"],
        device="cuda:0",
    )
    extractors = {
        "pytorch": original_extractor,
        "onnx_iobinding": adapter,
    }

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        root / split["dataset_root"], predictor.image_size
    )

    # Disk reads, image resizing and CPU batch creation are not timed.
    samples = [dataset[index] for index in range(len(dataset))]
    images = [sample["image"] for sample in samples]
    paths = [sample["path"] for sample in samples]

    rows = []
    results = {}

    print("GPU:", torch.cuda.get_device_name(0))
    print("Precision: IEEE FP32")
    print("Blur backend: original_2d")
    print("Warmup calls per variant:", args.warmup)
    print("Measured calls per variant:", args.iterations)

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
            for index in range(args.warmup):
                batch = batches[index % len(batches)]
                for name in extractors:
                    predictor.model.feature_extractor = extractors[name]
                    infer_to_cpu(predictor, batch)
            torch.cuda.synchronize()

            timings = {name: [] for name in extractors}
            print(f"Batch {batch_size}: paired measurements...", flush=True)

            for pair in range(args.iterations):
                batch_index = pair % len(batches)
                batch = batches[batch_index]

                # Alternate which implementation runs first.
                order = (
                    ("pytorch", "onnx_iobinding")
                    if pair % 2 == 0
                    else ("onnx_iobinding", "pytorch")
                )

                for position, name in enumerate(order):
                    # Backend selection is outside the timed region.
                    predictor.model.feature_extractor = extractors[name]
                    latency_ms = measure_ms(predictor, batch)
                    timings[name].append(latency_ms)
                    rows.append({
                        "batch_size": batch_size,
                        "pair": pair,
                        "position_in_pair": position,
                        "input_batch_index": batch_index,
                        "backend": name,
                        "latency_ms": latency_ms,
                    })

                if (pair + 1) % 25 == 0:
                    print(
                        f"  Pairs: {pair + 1}/{args.iterations}",
                        flush=True,
                    )

            batch_results = {
                name: summarize(values, batch_size)
                for name, values in timings.items()
            }
            speedup = (
                batch_results["pytorch"]["mean_batch_ms"]
                / batch_results["onnx_iobinding"]["mean_batch_ms"]
            )
            results[str(batch_size)] = {
                "variants": batch_results,
                "mean_latency_speedup": speedup,
            }

            for name, result in batch_results.items():
                print(
                    f"{name}: "
                    f"mean={result['mean_batch_ms']:.3f} ms | "
                    f"p50={result['p50_batch_ms']:.3f} ms | "
                    f"p95={result['p95_batch_ms']:.3f} ms | "
                    f"images/s={result['serial_images_per_second']:.2f}"
                )
            print(f"Mean latency speedup: {speedup:.3f}x")
    finally:
        predictor.model.feature_extractor = original_extractor

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "benchmarks" / f"onnx_ab_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "checkpoint_sha256": checkpoint_sha,
        "onnx_sha256": adapter.onnx_sha256,
        "parity_report_sha256": sha256_file(parity_path),
        "scope": (
            "Prepared CPU RGB batch -> H2D -> normalization -> "
            "PatchCore -> scores, labels and anomaly maps on CPU"
        ),
        "excluded": [
            "model loading", "disk I/O", "image decoding",
            "resizing", "CPU batch creation", "backend selection",
        ],
        "input_paths": paths,
        "batch_construction": (
            "Sequential batches; final batch wraps to the beginning "
            "to keep the requested batch size."
        ),
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("torch", "torchvision", "anomalib", "onnxruntime-gpu")
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
        "blur_backend": "original_2d",
        "warmup_calls_per_variant": args.warmup,
        "measured_calls_per_variant": args.iterations,
        "order": "paired, alternating AB/BA",
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

    shutil.copy2(parity_path, output_dir / "iobinding_parity.json")

    print("Output directory:", output_dir)
    print("PATCHCORE ONNX A/B BENCHMARK: OK")


if __name__ == "__main__":
    main()
