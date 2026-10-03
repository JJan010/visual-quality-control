import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from visual_quality.inference.filters import SeparableGaussianBlur2d
from visual_quality.inference.patchcore import PatchcorePredictor


def infer_to_cpu(predictor, images):
    result = predictor.predict_batch(images)
    return (
        result.scores.cpu(),
        result.labels.cpu(),
        result.anomaly_maps.cpu(),
    )


def summarize(times_ms, batch_size):
    values = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_batch_ms": float(values.mean()),
        "p50_batch_ms": float(np.percentile(values, 50)),
        "p95_batch_ms": float(np.percentile(values, 95)),
        "serial_images_per_second": float(
            batch_size * len(values) / (values.sum() / 1000.0)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("Warmup i iterations muszą być dodatnie.")

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    predictor = PatchcorePredictor(run_dir, device="cuda")
    generator = predictor.model.anomaly_map_generator
    original_blur = generator.blur

    parity = json.loads(
        (run_dir / "verification/separable_blur_parity.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        parity["status"] != "passed"
        or parity["checkpoint_sha256"] != predictor.checkpoint_sha256
    ):
        raise RuntimeError("Brak poprawnego testu zgodności dla tego modelu.")

    optimized_blur = SeparableGaussianBlur2d(
        original_blur.kernel
    ).to(predictor.device)
    optimized_blur.eval()

    variants = {
        "original_2d": original_blur,
        "separable_1d": optimized_blur,
    }

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset_root = project_root / split["dataset_root"]
    paths = sorted((dataset_root / "test").glob("*/*.png"))

    if len(paths) != 83:
        raise RuntimeError("Oczekiwano 83 zdjęć testowych.")

    prepared = []
    for path in paths:
        with Image.open(path) as image:
            prepared.append(predictor.prepare_image(image))

    summaries = []
    latency_rows = []

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Measured calls per variant:", args.iterations, flush=True)

    try:
        for batch_size in (1, 8):
            batches = [
                torch.stack([
                    prepared[(start + offset) % len(prepared)]
                    for offset in range(batch_size)
                ])
                for start in range(0, len(prepared), batch_size)
            ]

            print(f"\nBatch {batch_size}: warmup...", flush=True)
            for name, blur in variants.items():
                generator.blur = blur
                for index in range(args.warmup):
                    infer_to_cpu(predictor, batches[index % len(batches)])
            torch.cuda.synchronize()

            timings = {name: [] for name in variants}

            print(f"Batch {batch_size}: paired measurements...", flush=True)
            for index in range(args.iterations):
                images = batches[index % len(batches)]

                # Zmieniamy kolejność, zachowując ten sam batch w parze.
                order = (
                    ("original_2d", "separable_1d")
                    if index % 2 == 0
                    else ("separable_1d", "original_2d")
                )

                for position, name in enumerate(order, start=1):
                    generator.blur = variants[name]
                    torch.cuda.synchronize()

                    start = perf_counter()
                    outputs = infer_to_cpu(predictor, images)
                    torch.cuda.synchronize()
                    elapsed_ms = (perf_counter() - start) * 1000.0

                    del outputs
                    timings[name].append(elapsed_ms)
                    latency_rows.append({
                        "batch_size": batch_size,
                        "iteration": index + 1,
                        "variant": name,
                        "order_in_pair": position,
                        "latency_ms": elapsed_ms,
                    })

            statistics = {
                name: summarize(times, batch_size)
                for name, times in timings.items()
            }
            old_mean = statistics["original_2d"]["mean_batch_ms"]
            new_mean = statistics["separable_1d"]["mean_batch_ms"]
            speedup = old_mean / new_mean

            summaries.append({
                "batch_size": batch_size,
                "variants": statistics,
                "mean_latency_speedup": speedup,
            })

            for name, values in statistics.items():
                print(
                    f"{name}: "
                    f"mean={values['mean_batch_ms']:.3f} ms | "
                    f"p50={values['p50_batch_ms']:.3f} ms | "
                    f"p95={values['p95_batch_ms']:.3f} ms | "
                    f"images/s={values['serial_images_per_second']:.2f}"
                )
            print(f"Mean latency speedup: {speedup:.3f}x")

    finally:
        generator.blur = original_blur

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "benchmarks" / f"blur_ab_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "scope": "prepared CPU tensor to CPU scores, labels and anomaly maps",
        "execution": "paired sequential calls with alternating variant order",
        "warmup_calls_per_variant_and_batch_size": args.warmup,
        "measured_calls_per_variant_and_batch_size": args.iterations,
        "input_paths": [
            path.relative_to(dataset_root).as_posix() for path in paths
        ],
        "batch_policy": "sorted paths; final batch wraps to beginning",
        "input_memory": "pageable CPU memory, not pinned",
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "precision": "float32, no autocast",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "parity_check": parity,
        "results": summaries,
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "latencies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "batch_size", "iteration", "variant",
                "order_in_pair", "latency_ms",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(latency_rows)

    print("\nOutput directory:", output_dir)
    print("BLUR A/B BENCHMARK: OK")


if __name__ == "__main__":
    main()
