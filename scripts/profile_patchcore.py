import argparse
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile, record_function

from visual_quality.inference.patchcore import PatchcorePredictor


def labeled_call(function, label):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with record_function(label):
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def label_model_stages(model):
    # Zmieniamy wyłącznie wywołania w pamięci tego procesu.
    # Oryginalne funkcje nadal wykonują wszystkie obliczenia.
    targets = [
        (model.feature_extractor, "forward", "stage/feature_extractor"),
        (model.feature_pooler, "forward", "stage/feature_pooling"),
        (model, "generate_embedding", "stage/embedding"),
        (model, "nearest_neighbors", "stage/nearest_neighbors"),
        (model.anomaly_map_generator, "forward", "stage/anomaly_map"),
    ]
    originals = []

    try:
        for owner, attribute, label in targets:
            original = getattr(owner, attribute)
            originals.append((owner, attribute, original))
            setattr(owner, attribute, labeled_call(original, label))
        yield
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("Warmup i iterations muszą być dodatnie.")
    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise SystemExit("Profiler nie udostępnia aktywności CUDA.")

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    predictor = PatchcorePredictor(run_dir, device="cuda")
    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset_root = project_root / split["dataset_root"]

    # Stały zestaw: po dwa zdjęcia z każdej kategorii.
    paths = []
    for category in ("broken_large", "broken_small", "contamination", "good"):
        candidates = sorted((dataset_root / "test" / category).glob("*.png"))
        if len(candidates) < 2:
            raise RuntimeError(f"Za mało zdjęć w kategorii {category}.")
        paths.extend(candidates[:2])
    paths = paths[:args.batch_size]

    prepared = []
    for path in paths:
        with Image.open(path) as image:
            prepared.append(predictor.prepare_image(image))

    # Transfer wejścia odbywa się przed profilowaniem.
    images = torch.stack(prepared).to(predictor.device)

    print("Batch size:", args.batch_size, flush=True)
    print("Warmup...", flush=True)
    for _ in range(args.warmup):
        predictor.predict_batch(images)
    torch.cuda.synchronize()

    print("Profiling...", flush=True)
    with label_model_stages(predictor.model):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            for _ in range(args.iterations):
                with record_function("pipeline"):
                    result = predictor.predict_batch(images)
                del result
                torch.cuda.synchronize()

    events = profiler.key_averages()
    selected = {
        event.key: event
        for event in events
        if event.key == "pipeline" or event.key.startswith("stage/")
    }
    total_gpu_us = selected["pipeline"].device_time_total

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        run_dir / "profiles"
        / f"batch{args.batch_size}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    # Pełna tabela operatorów pozwoli później zajrzeć głębiej.
    table = events.table(sort_by="self_device_time_total", row_limit=30)
    (output_dir / "operators.txt").write_text(
        table + "\n", encoding="utf-8"
    )

    if total_gpu_us <= 0:
        raise RuntimeError(
            "Profiler nie zarejestrował czasu GPU. "
            f"Zapisano diagnostykę w {output_dir / 'operators.txt'}"
        )

    stage_results = []
    print("\nGPU operator time per inference batch:")
    for name in sorted(selected):
        event = selected[name]
        gpu_ms = event.device_time_total / args.iterations / 1000.0
        share = 100.0 * event.device_time_total / total_gpu_us
        calls_per_batch = event.count / args.iterations

        stage_results.append({
            "name": name,
            "gpu_operator_ms_per_batch": gpu_ms,
            "share_of_pipeline_gpu_operator_time_percent": share,
            "calls_per_batch": calls_per_batch,
        })
        print(
            f"{name:26s} | {gpu_ms:8.3f} ms"
            f" | {share:6.2f}% | calls/batch: {calls_per_batch:.1f}"
        )

    report = {
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "batch_size": args.batch_size,
        "warmup_calls": args.warmup,
        "profiled_calls": args.iterations,
        "input_paths": [
            path.relative_to(dataset_root).as_posix() for path in paths
        ],
        "input_and_output_device": "cuda",
        "measurement": "summed GPU operator durations attributed to each scope",
        "includes_profiler_overhead": True,
        "note": "pipeline contains all stage scopes; do not add it to them",
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "stages": stage_results,
    }
    (output_dir / "stages.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("\nOutput directory:", output_dir)
    print("PATCHCORE PROFILE: OK")


if __name__ == "__main__":
    main()
