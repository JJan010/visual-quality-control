"""Evaluate fixed-threshold PatchCore masks against MVTec bottle ground truth.

No threshold search or model fitting is performed. Run from the repository root.
"""

import argparse
import base64
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CALIBRATION = (
    "artifacts/runs/patchcore_20261003_135714_557111/"
    "localization_calibration/20261003_182936_190431/threshold.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator: int, denominator: int) -> float | None:
    # Undefined ratios are reported as null, not silently converted to 0 or 1.
    return numerator / denominator if denominator else None


def pixel_counts(predicted: np.ndarray, target: np.ndarray) -> dict:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("Expected masks with matching 2D shapes.")
    if predicted.dtype != np.bool_ or target.dtype != np.bool_:
        raise TypeError("Expected boolean masks.")
    return {
        "tp": int(np.count_nonzero(predicted & target)),
        "fp": int(np.count_nonzero(predicted & ~target)),
        "fn": int(np.count_nonzero(~predicted & target)),
        "tn": int(np.count_nonzero(~predicted & ~target)),
    }


def metrics(counts: dict) -> dict:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    return {
        **counts,
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "pixel_false_positive_rate": ratio(fp, fp + tn),
    }


def aggregate(rows: list[dict]) -> dict:
    counts = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")}
    return metrics(counts)


def self_test() -> None:
    target = np.array([[1, 1], [0, 0]], dtype=bool)
    predicted = np.array([[1, 0], [1, 0]], dtype=bool)
    result = metrics(pixel_counts(predicted, target))
    assert all(result[key] == 1 for key in ("tp", "fp", "fn", "tn"))
    assert math.isclose(result["iou"], 1 / 3)
    assert result["dice"] == result["precision"] == result["recall"] == 0.5
    perfect = metrics(pixel_counts(target, target))
    assert perfect["iou"] == perfect["dice"] == 1.0
    empty = np.zeros((2, 2), dtype=bool)
    absent = metrics(pixel_counts(empty, empty))
    assert absent["iou"] is None and absent["dice"] is None
    assert absent["precision"] is None and absent["recall"] is None
    assert absent["pixel_false_positive_rate"] == 0.0
    missed = metrics(pixel_counts(empty, target))
    assert missed["iou"] == missed["dice"] == missed["recall"] == 0.0
    assert missed["precision"] is None
    print("METRIC SELF-CHECK: OK")


def load_target(root: Path, path: Path, original_size: tuple, size: int):
    if path.parent.name == "good":
        return np.zeros((size, size), dtype=bool), "", ""
    mask_path = root / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    with Image.open(mask_path) as image:
        if image.size != original_size:
            raise ValueError(f"Ground-truth dimensions differ from the image: {mask_path}")
        gray = image.convert("L")
        values = set(np.unique(np.asarray(gray)))
        if not values.issubset({0, 255}) or 255 not in values:
            raise ValueError(f"Expected a non-empty binary defect mask: {mask_path}")
        # Explicit protocol: Pillow nearest-neighbor at the model input resolution.
        target = np.asarray(gray.resize((size, size), Image.Resampling.NEAREST)) > 0
    if not target.any():
        raise ValueError(f"Defect disappeared after mask resizing: {mask_path}")
    return target, mask_path.relative_to(root).as_posix(), sha256(mask_path)


def evaluate(args) -> None:
    # Heavy/GPU imports are unnecessary for --self-test.
    from visual_quality.api.localization import (
        create_localization,
        load_localization_calibration,
    )
    from visual_quality.inference.runtime import load_predictor

    root = args.dataset_root.resolve()
    paths = sorted((root / "test").glob("*/*.png"))
    expected = {"good": 20, "broken_large": 20, "broken_small": 22, "contamination": 21}
    actual = dict(Counter(path.parent.name for path in paths))
    if actual != expected:
        raise RuntimeError(f"Unexpected bottle test-set composition: {actual}")

    predictor = load_predictor(args.config)
    calibration = load_localization_calibration(
        args.calibration_file, predictor, args.config
    )
    size = predictor.image_size
    rows = []
    print("Runtime:", predictor.backend, "/", predictor.blur_backend)
    print("Localization threshold:", calibration.threshold)
    print("Image classification threshold:", predictor.threshold)
    print("Ground-truth resize: Pillow NEAREST; evaluation resolution:", (size, size))

    for index, path in enumerate(paths, 1):
        with Image.open(path) as image:
            original_size = image.size
            tensor = predictor.prepare_image(image)

        prediction = predictor.predict_batch(tensor.unsqueeze(0))
        maps = prediction.anomaly_maps.detach().cpu().numpy()
        if maps.shape != (1, 1, size, size) or not np.isfinite(maps).all():
            raise RuntimeError(f"Invalid model map: {path}")

        # Reuse the exact mask implementation used by the GUI/API.
        region = create_localization(maps[0, 0], calibration)
        with Image.open(BytesIO(base64.b64decode(region.mask_png_base64))) as image:
            predicted = np.asarray(image) > 0
        if int(predicted.sum()) != region.mask_pixels:
            raise RuntimeError("Decoded GUI mask differs from its recorded pixel count.")

        target, gt_path, gt_sha = load_target(root, path, original_size, size)
        counts = pixel_counts(predicted, target)
        score = float(prediction.scores.detach().cpu().item())
        image_label = bool(prediction.labels.detach().cpu().item())
        if not math.isfinite(score) or image_label != (score > predictor.threshold):
            raise RuntimeError("Invalid image-level result.")

        rows.append({
            "path": path.relative_to(root).as_posix(),
            "category": path.parent.name,
            "image_sha256": sha256(path),
            "ground_truth_path": gt_path,
            "ground_truth_sha256": gt_sha,
            "is_defective": path.parent.name != "good",
            "image_score": score,
            "image_predicted_anomaly": image_label,
            "has_predicted_region": bool(predicted.any()),
            "predicted_pixels": int(predicted.sum()),
            "ground_truth_pixels": int(target.sum()),
            **metrics(counts),
        })
        if index % 10 == 0 or index == len(paths):
            print(f"Evaluated image {index}/{len(paths)}")

    defective = [row for row in rows if row["is_defective"]]
    normal = [row for row in rows if not row["is_defective"]]
    micro = aggregate(rows)
    macro = {
        "images": len(defective),
        "mean_iou": float(np.mean([row["iou"] for row in defective])),
        "mean_dice": float(np.mean([row["dice"] for row in defective])),
    }
    false_region_count = sum(row["has_predicted_region"] for row in normal)
    classification = {
        "tp": sum(row["image_predicted_anomaly"] for row in defective),
        "fn": sum(not row["image_predicted_anomaly"] for row in defective),
        "fp": sum(row["image_predicted_anomaly"] for row in normal),
        "tn": sum(not row["image_predicted_anomaly"] for row in normal),
    }
    categories = {}
    for category in sorted(expected):
        selected = [row for row in rows if row["category"] == category]
        categories[category] = {
            "images": len(selected),
            "images_with_any_region": sum(row["has_predicted_region"] for row in selected),
            "micro_pixel_metrics": aggregate(selected),
            "macro_iou_defective_only": (
                float(np.mean([row["iou"] for row in selected])) if category != "good" else None
            ),
            "macro_dice_defective_only": (
                float(np.mean([row["dice"] for row in selected])) if category != "good" else None
            ),
        }

    # Detect accidental changes to the calibration file during evaluation.
    if sha256(args.calibration_file) != calibration.sha256:
        raise RuntimeError("Calibration file changed during evaluation.")
    report = {
        "status": "completed",
        "scope": "fixed-threshold mask evaluation; not a segmentation acceptance gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "calibration_sha256": calibration.sha256,
        "localization_threshold": calibration.threshold,
        "pixel_decision_rule": "anomaly_map > threshold",
        "threshold_tuned_on_test": False,
        "batch_size": 1,
        "evaluation_resolution": [size, size],
        "ground_truth_resize": "Pillow.Image.resize / Resampling.NEAREST, then > 0",
        "undefined_metric_policy": "null for zero denominator; empty/empty is not scored as perfect",
        "macro_policy": "equal-weight mean over the 63 defective images only",
        "micro_policy": "pooled pixel counts over all 83 test images, including normal-image FP",
        "versions": {name: version(name) for name in ("numpy", "pillow")},
        "runtime": predictor.runtime_metadata(),
        "test_images": len(rows),
        "total_pixels": sum(micro[key] for key in ("tp", "fp", "fn", "tn")),
        "defect_pixels": micro["tp"] + micro["fn"],
        "micro_all_test_pixels": micro,
        "macro_defective_images": macro,
        "normal_images": {
            "images": len(normal),
            "images_with_any_region": false_region_count,
            "region_alarm_rate": false_region_count / len(normal),
            "false_positive_pixels": sum(row["fp"] for row in normal),
        },
        "per_category": categories,
        "image_classification": classification,
        "test_set_note": "This test set has been reused for engineering checks; it is not a new holdout.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = predictor.run_dir / "localization_evaluation" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (directory / "calibration.json").write_bytes(args.calibration_file.read_bytes())
    with (directory / "per_image.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("\nPixel metrics — pooled over all 83 images:")
    for key in ("iou", "dice", "precision", "recall"):
        value = micro[key]
        print(f"  {key}: {value:.6f}" if value is not None else f"  {key}: undefined")
    print(f"Macro IoU — 63 defective images: {macro['mean_iou']:.6f}")
    print(f"Macro Dice — 63 defective images: {macro['mean_dice']:.6f}")
    print(f"Normal images with any region: {false_region_count}/{len(normal)} ({false_region_count / len(normal):.2%})")
    for category, stats in categories.items():
        print(f"{category}: regions {stats['images_with_any_region']}/{stats['images']}; pooled IoU={stats['micro_pixel_metrics']['iou']}")
    print("Image classification:", classification)
    print("Output directory:", directory)
    print("LOCALIZATION MASK EVALUATION: COMPLETED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime/patchcore_tensorrt.json"))
    parser.add_argument("--calibration-file", type=Path, default=Path(DEFAULT_CALIBRATION))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec_ad/bottle"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
