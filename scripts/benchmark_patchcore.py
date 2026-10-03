import argparse
import csv
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from visual_quality.inference.patchcore import PatchcorePredictor


def infer_to_cpu(predictor, images):
    result = predictor.predict_batch(images)
    return (
        result.scores.cpu(),
        result.labels.cpu(),
        result.anomaly_maps.cpu(),
    )


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
    device = predictor.device

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset_root = project_root / split["dataset_root"]
    paths = sorted((dataset_root / "test").glob("*/*.png"))

    if len(paths) != 83:
        raise RuntimeError(f"Oczekiwano 83 zdjęć, znaleziono {len(paths)}.")

    # Odczyt i przygotowanie zdjęć odbywają się poza pomiarem.
    prepared = []
    relative_paths = []
    for path in paths:
        with Image.open(path) as image:
            prepared.append(predictor.prepare_image(image))
        relative_paths.append(path.relative_to(dataset_root).as_posix())

    results = []
    latency_rows = []

    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    print("Warmup calls per batch size:", args.warmup, flush=True)
    print("Measured calls per batch size:", args.iterations, flush=True)

    for batch_size in (1, 8):
        # Tworzymy batche przed pomiarem. Ostatni uzupełniamy
        # obrazami z początku listy, aby zachować stały rozmiar.
        index_groups = [
            [(start + offset) % len(prepared) for offset in range(batch_size)]
            for start in range(0, len(prepared), batch_size)
        ]
        batches = [
            torch.stack([prepared[index] for index in indices])
            for indices in index_groups
        ]

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

        print(f"Batch {batch_size}: warmup...", flush=True)
        for index in range(args.warmup):
            batch = batches[index % len(batches)]
            infer_to_cpu(predictor, batch)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        print(f"Batch {batch_size}: measuring...", flush=True)
        durations_ms = []

        for index in range(args.iterations):
            batch = batches[index % len(batches)]

            # Przed startem nie może pozostać wcześniejsza praca GPU.
            torch.cuda.synchronize(device)
            start = perf_counter()

            cpu_outputs = infer_to_cpu(predictor, batch)
            torch.cuda.synchronize(device)

            elapsed_ms = (perf_counter() - start) * 1000.0
            durations_ms.append(elapsed_ms)
            del cpu_outputs

            latency_rows.append({
                "batch_size": batch_size,
                "iteration": index + 1,
                "latency_ms": elapsed_ms,
            })

        times = np.asarray(durations_ms, dtype=np.float64)
        result = {
            "batch_size": batch_size,
            "warmup_calls": args.warmup,
            "measured_calls": args.iterations,
            "mean_batch_ms": float(times.mean()),
            "p50_batch_ms": float(np.percentile(times, 50)),
            "p95_batch_ms": float(np.percentile(times, 95)),
            "serial_images_per_second": float(
                batch_size * args.iterations / (times.sum() / 1000.0)
            ),
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024**2
            ),
            "input_batches": [
                [relative_paths[index] for index in indices]
                for indices in index_groups
            ],
        }
        results.append(result)

        print(f"Batch size: {batch_size}")
        print(f"Mean batch latency: {result['mean_batch_ms']:.3f} ms")
        print(f"p50 batch latency: {result['p50_batch_ms']:.3f} ms")
        print(f"p95 batch latency: {result['p95_batch_ms']:.3f} ms")
        print(f"Serial images/s: {result['serial_images_per_second']:.2f}")
        print(f"Peak allocated: {result['peak_allocated_mib']:.2f} MiB")
        print(f"Peak reserved: {result['peak_reserved_mib']:.2f} MiB")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "benchmarks" / f"pytorch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "scope": "prepared CPU tensor to CPU scores, labels and anomaly maps",
        "excluded": [
            "model loading",
            "file reading and decoding",
            "image resizing and conversion to tensor",
            "batch assembly",
            "HTTP, queueing and serialization",
        ],
        "execution": "sequential calls, no overlap between batches",
        "input_memory": "pageable CPU memory, not pinned",
        "precision": "float32, no autocast",
        "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "torchvision": version("torchvision"),
        "anomalib": version("anomalib"),
        "timm": version("timm"),
        "cuda_runtime": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "results": results,
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "latencies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["batch_size", "iteration", "latency_ms"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(latency_rows)

    print("Output directory:", output_dir)
    print("PATCHCORE BENCHMARK: OK")


if __name__ == "__main__":
    main()
