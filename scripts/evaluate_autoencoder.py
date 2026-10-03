import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from visual_quality.models.autoencoder import ConvAutoencoder


def safe_divide(numerator, denominator):
    # None oznacza, że miara jest niezdefiniowana.
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    calibration = json.loads(
        (run_dir / "calibration/threshold.json").read_text(encoding="utf-8")
    )
    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    # Próg musi odpowiadać ocenianemu modelowi i sposobowi liczenia błędu.
    if calibration["checkpoint_epoch"] != checkpoint["epoch"]:
        raise ValueError("Próg pochodzi z innej epoki.")
    if calibration["image_size"] != config["image_size"]:
        raise ValueError("Niezgodny rozmiar obrazu.")
    if calibration["score"] != "mean_squared_reconstruction_error":
        raise ValueError("Nieobsługiwana definicja wyniku anomalii.")
    if calibration["decision_rule"] != "score > threshold":
        raise ValueError("Nieobsługiwana reguła decyzji.")

    threshold = float(calibration["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Próg musi być skończoną liczbą.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA niedostępna. Sprawdź aktywne środowisko.")

    device = torch.device("cuda")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Identyczne przetwarzanie obrazu jak przy treningu i kalibracji.
    image_size = config["image_size"]
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset_root = project_root / split["dataset_root"]
    paths = sorted((dataset_root / "test").glob("*/*.png"))

    expected_counts = {
        "good": 20,
        "broken_large": 20,
        "broken_small": 22,
        "contamination": 21,
    }
    actual_counts = dict(Counter(path.parent.name for path in paths))
    if actual_counts != expected_counts:
        raise ValueError(f"Nieoczekiwany skład zbioru: {actual_counts}")

    records = []
    batch_size = config["batch_size"]

    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            tensors = []

            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))

            images = torch.stack(tensors).to(device)
            reconstructions = model(images)

            if reconstructions.shape != images.shape:
                raise ValueError("Niezgodne kształty rekonstrukcji i wejścia.")

            scores = (images - reconstructions).square().mean(dim=(1, 2, 3))
            if not torch.isfinite(scores).all().item():
                raise ValueError("Wyniki zawierają NaN lub Inf.")

            for path, score in zip(batch_paths, scores.cpu().tolist()):
                category = path.parent.name
                records.append({
                    "path": path.relative_to(dataset_root).as_posix(),
                    "category": category,
                    "is_anomaly": int(category != "good"),
                    "score": score,
                    "predicted_anomaly": int(score > threshold),
                })

    labels = np.array([row["is_anomaly"] for row in records], dtype=bool)
    predictions = np.array(
        [row["predicted_anomaly"] for row in records], dtype=bool
    )

    tp = int((labels & predictions).sum())
    fn = int((labels & ~predictions).sum())
    fp = int((~labels & predictions).sum())
    tn = int((~labels & ~predictions).sum())

    metrics = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold": threshold,
        "score": calibration["score"],
        "decision_rule": calibration["decision_rule"],
        "num_images": len(records),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": safe_divide(tp, tp + fn),
        "precision": safe_divide(tp, tp + fp),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
    }

    per_category = {}
    for category in sorted(expected_counts):
        group = [row for row in records if row["category"] == category]
        alarms = sum(row["predicted_anomaly"] for row in group)
        per_category[category] = {
            "images": len(group),
            "alarms": alarms,
            "alarm_rate": alarms / len(group),
        }
    metrics["per_category"] = per_category

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "test_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Test images: {len(records)}")
    print(f"Fixed threshold: {threshold:.10f}")
    print(f"TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")

    for name in ["recall", "precision", "false_positive_rate", "f1"]:
        value = metrics[name]
        display = "undefined" if value is None else f"{value:.2%}"
        print(f"{name}: {display}")

    for category, result in per_category.items():
        print(
            f"{category}: alarms "
            f"{result['alarms']}/{result['images']} "
            f"({result['alarm_rate']:.2%})"
        )

    print(f"Output directory: {output_dir}")
    print("EVALUATION: OK")


if __name__ == "__main__":
    main()
