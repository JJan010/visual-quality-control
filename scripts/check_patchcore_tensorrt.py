import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import tensorrt as trt
from torch.utils.data import DataLoader

from check_patchcore_onnx_iobinding import check_close, ATOL, RTOL
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
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    export_dir = args.export_dir.resolve()

    torch.cuda.set_device(0)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"

    predictor = PatchcorePredictor(
        run_dir, device="cuda:0", blur_backend="original_2d"
    )
    torch_extractor = predictor.model.feature_extractor
    trt_extractor = TensorRTCudaFeatureExtractor(args.engine_dir)

    checkpoint_sha = sha256_file(run_dir / "model.pt")
    if checkpoint_sha != trt_extractor.report["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the engine build.")

    onnx_extractor = OnnxCudaFeatureExtractor(
        export_dir / "feature_extractor.onnx",
        expected_sha256=trt_extractor.report["onnx_sha256"],
        device="cuda:0",
    )

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        root / split["dataset_root"], predictor.image_size
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

    results = []

    try:
        for batch_size in (1, 8):
            loader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, num_workers=0,
            )
            comparisons = {
                name: {
                    "max_score_difference": 0.0,
                    "max_map_difference": 0.0,
                    "changed_decisions": 0,
                }
                for name in ("pytorch", "onnx")
            }
            checked_paths = []
            changed_vs_saved = 0

            for index, batch in enumerate(loader, start=1):
                predictor.model.feature_extractor = trt_extractor
                candidate = predictor.predict_batch(batch["image"])

                for name, extractor in (
                    ("pytorch", torch_extractor),
                    ("onnx", onnx_extractor),
                ):
                    predictor.model.feature_extractor = extractor
                    reference = predictor.predict_batch(batch["image"])

                    try:
                        score_difference = check_close(
                            candidate.scores, reference.scores
                        )
                        map_difference = check_close(
                            candidate.anomaly_maps, reference.anomaly_maps
                        )
                    except AssertionError as error:
                        raise AssertionError(
                            f"TensorRT vs {name}, batch size {batch_size}, "
                            f"batch {index}, files: {batch['path']}\n{error}"
                        ) from error

                    item = comparisons[name]
                    item["max_score_difference"] = max(
                        item["max_score_difference"], score_difference
                    )
                    item["max_map_difference"] = max(
                        item["max_map_difference"], map_difference
                    )
                    item["changed_decisions"] += int(
                        (candidate.labels != reference.labels).sum().item()
                    )

                saved_labels = torch.tensor(
                    [
                        saved_scores[path] > predictor.threshold
                        for path in batch["path"]
                    ],
                    device="cuda:0",
                    dtype=torch.bool,
                )
                changed_vs_saved += int(
                    (
                        candidate.labels.reshape(-1) != saved_labels
                    ).sum().item()
                )
                checked_paths.extend(batch["path"])

                if index % 10 == 0 or index == len(loader):
                    print(
                        f"Batch size {batch_size}: "
                        f"compared batch {index}/{len(loader)}",
                        flush=True,
                    )

            if len(checked_paths) != 83:
                raise RuntimeError("Expected 83 images.")
            if len(set(checked_paths)) != len(checked_paths):
                raise RuntimeError("Duplicate checked images.")
            if set(checked_paths) != set(saved_scores):
                raise RuntimeError("Image set differs from saved baseline.")
            if changed_vs_saved or any(
                item["changed_decisions"] for item in comparisons.values()
            ):
                raise RuntimeError("Anomaly decisions changed.")

            result = {
                "batch_size": batch_size,
                "checked_images": len(checked_paths),
                "comparisons": comparisons,
                "changed_decisions_vs_saved": changed_vs_saved,
            }
            results.append(result)
            print(json.dumps(result, indent=2))
    finally:
        predictor.model.feature_extractor = torch_extractor

    report = {
        "status": "passed",
        "checkpoint_sha256": checkpoint_sha,
        "onnx_sha256": onnx_extractor.onnx_sha256,
        "engine_sha256": trt_extractor.engine_sha256,
        "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "torch_precision": "ieee",
        "ort_use_tf32": False,
        "tensorrt_precision": "fp32_tf32_off",
        "blur_backend": "original_2d",
        "threshold": predictor.threshold,
        "atol": ATOL,
        "rtol": RTOL,
        "execution": "dedicated CUDA stream, synchronized before return",
        "results": results,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "verification" / f"tensorrt_fp32_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Output directory:", output_dir)
    print("PATCHCORE TENSORRT FP32: OK")


if __name__ == "__main__":
    main()
