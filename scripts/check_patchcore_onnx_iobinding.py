import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.onnx_features import (
    OnnxCudaFeatureExtractor,
    sha256_file,
)
from visual_quality.inference.patchcore import PatchcorePredictor


ATOL = 1e-4
RTOL = 1e-4


def check_close(actual, reference):
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("Candidate contains NaN or Inf.")
    if not torch.isfinite(reference).all().item():
        raise RuntimeError("Reference contains NaN or Inf.")

    torch.testing.assert_close(
        actual, reference, atol=ATOL, rtol=RTOL
    )
    return float((actual - reference).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    export_dir = args.export_dir.resolve()

    summary = json.loads(
        (
            root / "docs/experiments/patchcore_onnx_export"
            / "validation_summary.json"
        ).read_text(encoding="utf-8")
    )
    if summary["status"] != "passed_for_tested_pipeline_configuration":
        raise RuntimeError("Missing reviewed pipeline validation.")
    if sha256_file(run_dir / "model.pt") != summary["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint does not match the validation.")

    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"

    predictor = PatchcorePredictor(
        run_dir, device="cuda:0", blur_backend="original_2d"
    )
    if predictor.threshold != summary["threshold"]:
        raise RuntimeError("Threshold differs from the reviewed experiment.")

    original_extractor = predictor.model.feature_extractor
    adapter = OnnxCudaFeatureExtractor(
        export_dir / "feature_extractor.onnx",
        expected_sha256=summary["onnx_sha256"],
        device="cuda:0",
    )

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset_root = root / split["dataset_root"]

    images = []
    for relative_path in split["validation"][:8]:
        with Image.open(dataset_root / relative_path) as image:
            images.append(predictor.prepare_image(image))
    if len(images) != 8:
        raise RuntimeError("Expected eight validation images.")

    normalized = predictor.normalize(
        torch.stack(images).to("cuda:0")
    ).contiguous()

    binding_checks = []
    for batch_size in (1, 3, 8):
        batch = normalized[:batch_size]
        bound_outputs = adapter(batch)

        # CPU transfers are intentional here: this is the reference path.
        ordinary_outputs = adapter.session.run(
            ["layer2", "layer3"],
            {"images": batch.cpu().numpy()},
        )

        for name, array in zip(
            ("layer2", "layer3"), ordinary_outputs
        ):
            reference = torch.from_numpy(array).to("cuda:0")
            difference = check_close(bound_outputs[name], reference)
            binding_checks.append({
                "batch_size": batch_size,
                "layer": name,
                "max_difference": difference,
            })
            print(
                f"I/O Binding | B={batch_size} | {name} | "
                f"max diff={difference:.10f}"
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

    dataset = BottleTestDataset(dataset_root, predictor.image_size)
    pipeline_checks = []

    try:
        for batch_size in (1, 8):
            loader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, num_workers=0,
            )
            checked_paths = []
            max_score_difference = 0.0
            max_map_difference = 0.0
            changed_decisions = 0
            changed_vs_saved = 0

            for index, batch in enumerate(loader, start=1):
                predictor.model.feature_extractor = original_extractor
                reference = predictor.predict_batch(batch["image"])

                predictor.model.feature_extractor = adapter
                candidate = predictor.predict_batch(batch["image"])

                max_score_difference = max(
                    max_score_difference,
                    check_close(candidate.scores, reference.scores),
                )
                max_map_difference = max(
                    max_map_difference,
                    check_close(
                        candidate.anomaly_maps, reference.anomaly_maps
                    ),
                )

                changed_decisions += int(
                    (candidate.labels != reference.labels).sum().item()
                )
                saved_labels = torch.tensor(
                    [
                        saved_scores[path] > predictor.threshold
                        for path in batch["path"]
                    ],
                    device=candidate.labels.device,
                    dtype=torch.bool,
                )
                changed_vs_saved += int(
                    (
                        candidate.labels.reshape(-1)
                        != saved_labels
                    ).sum().item()
                )
                checked_paths.extend(batch["path"])

                if index % 10 == 0 or index == len(loader):
                    print(
                        f"Pipeline B={batch_size}: "
                        f"batch {index}/{len(loader)}",
                        flush=True,
                    )

            if len(checked_paths) != 83:
                raise RuntimeError("Expected 83 checked images.")
            if len(set(checked_paths)) != len(checked_paths):
                raise RuntimeError("Duplicate checked images.")
            if set(checked_paths) != set(saved_scores):
                raise RuntimeError("Image set differs from saved baseline.")
            if changed_decisions or changed_vs_saved:
                raise RuntimeError("Anomaly decisions changed.")

            result = {
                "batch_size": batch_size,
                "checked_images": len(checked_paths),
                "max_score_difference": max_score_difference,
                "max_map_difference": max_map_difference,
                "changed_decisions": changed_decisions,
                "changed_decisions_vs_saved": changed_vs_saved,
            }
            pipeline_checks.append(result)
            print(json.dumps(result, indent=2))
    finally:
        predictor.model.feature_extractor = original_extractor

    report = {
        "status": "passed",
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "onnx_sha256": adapter.onnx_sha256,
        "gpu": torch.cuda.get_device_name(0),
        "torch_precision": "ieee",
        "ort_use_tf32": False,
        "blur_backend": "original_2d",
        "threshold": predictor.threshold,
        "atol": ATOL,
        "rtol": RTOL,
        "binding_checks": binding_checks,
        "pipeline_checks": pipeline_checks,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "verification" / f"onnx_iobinding_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Output directory:", output_dir)
    print("PATCHCORE ONNX I/O BINDING: OK")


if __name__ == "__main__":
    main()
