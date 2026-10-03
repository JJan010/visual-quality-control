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

from benchmark_patchcore_onnx import infer_to_cpu, measure_ms, summarize
from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.backends import configure_ieee_fp32
from visual_quality.inference.patchcore import PatchcorePredictor, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    parity_path = args.parity_report.resolve()

    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity["status"] != "passed":
        raise RuntimeError("A successful parity report is required.")

    checks = {item["batch_size"]: item for item in parity["results"]}
    for batch_size in (1, 8):
        item = checks[batch_size]
        if item["checked_images"] != 83:
            raise RuntimeError("Incomplete parity check.")
        for key in (
            "changed_decisions_vs_pytorch",
            "changed_decisions_vs_trt_original",
            "changed_decisions_vs_saved",
            "max_score_difference_vs_trt_original",
        ):
            if item[key] != 0:
                raise RuntimeError(f"Parity requirement failed: {key}")

    torch.cuda.set_device(0)
    configure_ieee_fp32()

    common = {
        "run_dir": run_dir,
        "device": "cuda:0",
        "precision": "ieee",
    }
    predictors = {
        "pytorch_original": PatchcorePredictor(
            **common,
            backend="pytorch",
            blur_backend="original_2d",
        ),
        "tensorrt_original": PatchcorePredictor(
            **common,
            backend="tensorrt",
            engine_dir=args.engine_dir,
            blur_backend="original_2d",
        ),
        "tensorrt_separable": PatchcorePredictor(
            **common,
            backend="tensorrt",
            engine_dir=args.engine_dir,
            blur_backend="separable_1d",
        ),
    }

    runtime = {
        name: predictor.runtime_metadata()
        for name, predictor in predictors.items()
    }
    for name, metadata in runtime.items():
        if metadata != parity["runtime"][name]:
            raise RuntimeError(
                f"Runtime configuration differs from parity check: {name}"
            )

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        root / split["dataset_root"],
        predictors["pytorch_original"].image_size,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    images = [sample["image"] for sample in samples]

    warmup = 30
    iterations = 120
    seed = 42

    orders = list(itertools.permutations(predictors)) * (iterations // 6)
    random.Random(seed).shuffle(orders)

    rows = []
    results = {}

    print("GPU:", torch.cuda.get_device_name(0))
    print("Precision: IEEE FP32, TF32 disabled")
    print("Warmup calls per configuration:", warmup)
    print("Measured calls per configuration:", iterations)

    for batch_size in (1, 8):
        # Prepare full-sized CPU batches before timing.
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
                infer_to_cpu(predictors[name], batch)
        torch.cuda.synchronize()

        timings = {name: [] for name in predictors}
        print(f"Batch {batch_size}: measuring...", flush=True)

        for round_index, order in enumerate(orders):
            batch_index = round_index % len(batches)
            batch = batches[batch_index]

            for position, name in enumerate(order):
                latency_ms = measure_ms(predictors[name], batch)
                timings[name].append(latency_ms)
                rows.append({
                    "batch_size": batch_size,
                    "round": round_index,
                    "position": position,
                    "input_batch_index": batch_index,
                    "configuration": name,
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

        baseline_ms = variants["pytorch_original"]["mean_batch_ms"]
        trt_original_ms = variants["tensorrt_original"]["mean_batch_ms"]

        for name, item in variants.items():
            item["speedup_vs_pytorch_original"] = (
                baseline_ms / item["mean_batch_ms"]
            )
            item["speedup_vs_tensorrt_original"] = (
                trt_original_ms / item["mean_batch_ms"]
            )
            print(
                f"{name}: "
                f"mean={item['mean_batch_ms']:.3f} ms | "
                f"p50={item['p50_batch_ms']:.3f} ms | "
                f"p95={item['p95_batch_ms']:.3f} ms | "
                f"images/s={item['serial_images_per_second']:.2f} | "
                f"speedup vs PyTorch="
                f"{item['speedup_vs_pytorch_original']:.3f}x"
            )

        extra_speedup = variants["tensorrt_separable"][
            "speedup_vs_tensorrt_original"
        ]
        print(f"Separable blur speedup within TensorRT: {extra_speedup:.3f}x")
        results[str(batch_size)] = variants

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "benchmarks" / f"optimized_pipeline_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "runtime": runtime,
        "parity_report_sha256": file_sha256(parity_path),
        "benchmark_script_sha256": file_sha256(Path(__file__)),
        "timing_helpers_sha256": file_sha256(
            root / "scripts/benchmark_patchcore_onnx.py"
        ),
        "scope": (
            "Prepared CPU RGB batch -> H2D -> normalization -> "
            "complete PatchCore -> scores, labels and maps on CPU"
        ),
        "excluded": [
            "model loading", "disk I/O", "decoding",
            "resizing", "CPU batch creation",
        ],
        "input_paths": [sample["path"] for sample in samples],
        "batch_construction": "Sequential; final batch wraps to the start.",
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("torch", "torchvision", "anomalib", "tensorrt-cu12")
        },
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "warmup_calls_per_configuration": warmup,
        "measured_calls_per_configuration": iterations,
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

    shutil.copy2(parity_path, output_dir / "optimized_parity.json")
    print("Output directory:", output_dir)
    print("OPTIMIZED PATCHCORE BENCHMARK: OK")


if __name__ == "__main__":
    main()
