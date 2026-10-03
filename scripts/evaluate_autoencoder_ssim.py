import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from visual_quality.models.autoencoder import ConvAutoencoder
from visual_quality.scoring.ssim import SSIMScorer


def save_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_scores(path, records, threshold):
    rows = [
        {
            **record,
            "predicted_anomaly": int(record["score"] > threshold),
        }
        for record in records
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calculate_scores(paths, dataset_root, transform, model, scorer, batch_size):
    records = []

    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            tensors = []

            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))

            images = torch.stack(tensors).to("cuda")
            reconstructions = model(images)
            scores, _ = scorer(images, reconstructions)

            if not torch.isfinite(scores).all().item():
                raise ValueError("Wyniki SSIM zawierają NaN lub Inf.")

            for path, score in zip(batch_paths, scores.cpu().tolist()):
                records.append({
                    "path": path.relative_to(dataset_root).as_posix(),
                    "category": path.parent.name,
                    "is_anomaly": int(path.parent.name != "good"),
                    "score": score,
                })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA niedostępna.")

    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset_root = project_root / split["dataset_root"]

    model = ConvAutoencoder().to("cuda").eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    scorer = SSIMScorer().to("cuda").eval()

    size = config["image_size"]
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    validation_paths = [dataset_root / path for path in split["validation"]]
    if len(validation_paths) != 42 or len(set(validation_paths)) != 42:
        raise ValueError("Oczekiwano 42 różnych zdjęć walidacyjnych.")
    if any(path.parent.name != "good" for path in validation_paths):
        raise ValueError("Kalibracja wymaga wyłącznie prawidłowych zdjęć.")

    output_dir = run_dir / "scoring_ssim_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "score": "mean_one_minus_ssim",
        "image_size": size,
        "window_size": scorer.window_size,
        "sigma": scorer.sigma,
        "c1": scorer.c1,
        "c2": scorer.c2,
        "data_range": 1.0,
        "channels": "RGB, channel-wise SSIM followed by mean",
        "border": "valid windows, no padding",
        "spatial_aggregation": "mean",
        "torch_version": str(torch.__version__),
    }
    save_json(output_dir / "settings.json", settings)

    # Etap 1: kalibracja. Zbiór testowy nie uczestniczy w wyborze progu.
    validation = calculate_scores(
        validation_paths, dataset_root, transform,
        model, scorer, config["batch_size"],
    )
    validation_scores = np.array([row["score"] for row in validation])
    threshold = float(
        np.quantile(validation_scores, 0.95, method="higher")
    )
    validation_alarms = int((validation_scores > threshold).sum())

    calibration = {
        **settings,
        "split": "validation",
        "num_images": len(validation),
        "quantile": 0.95,
        "quantile_method": "higher",
        "threshold": threshold,
        "decision_rule": "score > threshold",
        "false_alarms_on_calibration": validation_alarms,
        "independent_of_checkpoint_selection": False,
    }
    save_json(output_dir / "threshold.json", calibration)
    save_scores(output_dir / "validation_scores.csv", validation, threshold)

    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Validation images: {len(validation)}")
    print(f"SSIM threshold: {threshold:.8f}")
    print(f"Calibration alarms: {validation_alarms}/{len(validation)}")

    # Etap 2: ocena testu z już ustalonym progiem.
    test_paths = sorted((dataset_root / "test").glob("*/*.png"))
    counts = dict(Counter(path.parent.name for path in test_paths))
    expected = {
        "good": 20,
        "broken_large": 20,
        "broken_small": 22,
        "contamination": 21,
    }
    if counts != expected:
        raise ValueError(f"Nieoczekiwany skład testu: {counts}")

    test = calculate_scores(
        test_paths, dataset_root, transform,
        model, scorer, config["batch_size"],
    )
    scores = np.array([row["score"] for row in test])
    labels = np.array([row["is_anomaly"] for row in test], dtype=bool)
    predictions = scores > threshold

    tp = int((labels & predictions).sum())
    fn = int((labels & ~predictions).sum())
    fp = int((~labels & predictions).sum())
    tn = int((~labels & ~predictions).sum())

    # AUROC przez porównanie wszystkich par: wada kontra obraz prawidłowy.
    positive_scores = scores[labels][:, None]
    negative_scores = scores[~labels][None, :]
    auroc = float(
        (
            (positive_scores > negative_scores)
            + 0.5 * (positive_scores == negative_scores)
        ).mean()
    )

    per_category = {}
    for category in sorted(expected):
        group = [row for row in test if row["category"] == category]
        alarms = sum(row["score"] > threshold for row in group)
        per_category[category] = {
            "images": len(group),
            "alarms": alarms,
            "alarm_rate": alarms / len(group),
        }

    metrics = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "score": settings["score"],
        "threshold": threshold,
        "num_images": len(test),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / (fp + tn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "image_auroc": auroc,
        "per_category": per_category,
    }
    save_json(output_dir / "metrics.json", metrics)
    save_scores(output_dir / "test_scores.csv", test, threshold)

    print(f"Test images: {len(test)}")
    print(f"TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
    for name in ["recall", "precision", "false_positive_rate", "f1"]:
        value = metrics[name]
        display = "undefined" if value is None else f"{value:.2%}"
        print(f"{name}: {display}")
    print(f"Image AUROC: {auroc:.6f}")

    for category, result in per_category.items():
        print(f"{category}: alarms {result['alarms']}/{result['images']}")

    print("Device: cuda")
    print(f"Output directory: {output_dir}")
    print("SSIM EVALUATION: OK")


if __name__ == "__main__":
    main()
