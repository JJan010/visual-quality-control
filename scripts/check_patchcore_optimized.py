import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from check_patchcore_onnx_iobinding import check_close, ATOL, RTOL
from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.backends import configure_ieee_fp32
from visual_quality.inference.patchcore import PatchcorePredictor


BLUR_ATOL = 1e-4
BLUR_RTOL = 1e-5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()

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

    for name, predictor in predictors.items():
        metadata = predictor.runtime_metadata()
        print(
            f"{name}: "
            f"{metadata['feature_extractor_class']} + "
            f"{metadata['blur_class']}"
        )

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        root / split["dataset_root"],
        predictors["pytorch_original"].image_size,
    )

    with (run_dir / "evaluation/test_scores.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        saved_rows = list(csv.DictReader(handle))

    saved_scores = {
        row["path"]: float(row["score"]) for row in saved_rows
    }
    if len(saved_scores) != len(saved_rows):
        raise RuntimeError("Duplicate paths in saved baseline.")

    threshold = predictors["tensorrt_separable"].threshold
    results = []

    for batch_size in (1, 8):
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, num_workers=0,
        )
        stats = {
            "batch_size": batch_size,
            "max_score_difference_vs_pytorch": 0.0,
            "max_map_difference_vs_pytorch": 0.0,
            "max_score_difference_vs_trt_original": 0.0,
            "max_map_difference_vs_trt_original": 0.0,
            "changed_decisions_vs_pytorch": 0,
            "changed_decisions_vs_trt_original": 0,
            "changed_decisions_vs_saved": 0,
        }
        checked_paths = []

        for index, batch in enumerate(loader, start=1):
            reference = predictors["pytorch_original"].predict_batch(
                batch["image"]
            )
            control = predictors["tensorrt_original"].predict_batch(
                batch["image"]
            )
            candidate = predictors["tensorrt_separable"].predict_batch(
                batch["image"]
            )

            # Combined backend + blur change versus the PyTorch reference.
            score_difference = check_close(
                candidate.scores, reference.scores
            )
            map_difference = check_close(
                candidate.anomaly_maps, reference.anomaly_maps
            )
            stats["max_score_difference_vs_pytorch"] = max(
                stats["max_score_difference_vs_pytorch"],
                score_difference,
            )
            stats["max_map_difference_vs_pytorch"] = max(
                stats["max_map_difference_vs_pytorch"],
                map_difference,
            )

            # Isolate the blur change within the TensorRT backend.
            torch.testing.assert_close(
                candidate.scores, control.scores,
                atol=0, rtol=0,
            )

            if not torch.isfinite(control.anomaly_maps).all().item():
                raise RuntimeError("Control map contains NaN or Inf.")

            torch.testing.assert_close(
                candidate.anomaly_maps,
                control.anomaly_maps,
                atol=BLUR_ATOL,
                rtol=BLUR_RTOL,
            )

            stats["max_score_difference_vs_trt_original"] = max(
                stats["max_score_difference_vs_trt_original"],
                float((candidate.scores - control.scores).abs().max().item()),
            )
            stats["max_map_difference_vs_trt_original"] = max(
                stats["max_map_difference_vs_trt_original"],
                float(
                    (candidate.anomaly_maps - control.anomaly_maps)
                    .abs().max().item()
                ),
            )

            stats["changed_decisions_vs_pytorch"] += int(
                (candidate.labels != reference.labels).sum().item()
            )
            stats["changed_decisions_vs_trt_original"] += int(
                (candidate.labels != control.labels).sum().item()
            )

            saved_labels = torch.tensor(
                [
                    saved_scores[path] > threshold
                    for path in batch["path"]
                ],
                dtype=torch.bool,
                device="cuda:0",
            )
            stats["changed_decisions_vs_saved"] += int(
                (
                    candidate.labels.reshape(-1) != saved_labels
                ).sum().item()
            )

            checked_paths.extend(batch["path"])
            if index % 10 == 0 or index == len(loader):
                print(
                    f"B={batch_size}: batch {index}/{len(loader)}",
                    flush=True,
                )

        if len(checked_paths) != 83:
            raise RuntimeError("Expected 83 images.")
        if len(set(checked_paths)) != len(checked_paths):
            raise RuntimeError("Duplicate checked images.")
        if set(checked_paths) != set(saved_scores):
            raise RuntimeError("Image set differs from saved baseline.")

        for key in (
            "changed_decisions_vs_pytorch",
            "changed_decisions_vs_trt_original",
            "changed_decisions_vs_saved",
        ):
            if stats[key] != 0:
                raise RuntimeError(f"Decision mismatch: {key}")

        stats["checked_images"] = len(checked_paths)
        results.append(stats)
        print(json.dumps(stats, indent=2))

    report = {
        "status": "passed",
        "scope": "TensorRT FP32 combined with separable Gaussian blur",
        "pipeline_tolerance": {"atol": ATOL, "rtol": RTOL},
        "blur_map_tolerance": {"atol": BLUR_ATOL, "rtol": BLUR_RTOL},
        "blur_score_tolerance": {"atol": 0, "rtol": 0},
        "runtime": {
            name: predictor.runtime_metadata()
            for name, predictor in predictors.items()
        },
        "results": results,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "verification" / f"optimized_pipeline_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Output directory:", output_dir)
    print("OPTIMIZED PATCHCORE PIPELINE: OK")


if __name__ == "__main__":
    main()
