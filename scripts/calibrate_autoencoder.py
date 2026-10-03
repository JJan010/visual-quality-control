import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from visual_quality.data.bottle import BottleNormalDataset
from visual_quality.models.autoencoder import ConvAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    # Sprawdzamy, czy używamy podziału danych z tego eksperymentu.
    saved_split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    current_split = json.loads(
        (project_root / "configs/splits/bottle_v1.json").read_text(
            encoding="utf-8"
        )
    )
    if saved_split != current_split:
        raise ValueError("Podział danych różni się od zapisanego w eksperymencie.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA niedostępna. Sprawdź aktywne środowisko.")

    device = torch.device("cuda")
    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    # Wczytujemy wyuczone wagi. Nie uruchamiamy ponownego treningu.
    model = ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = BottleNormalDataset(
        project_root=project_root,
        split="validation",
        image_size=config["image_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    paths = []
    score_values = []

    # Inferencja: obliczenia bez gradientów i aktualizacji wag.
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            reconstructions = model(images)

            if reconstructions.shape != images.shape:
                raise ValueError("Niezgodne kształty wejścia i rekonstrukcji.")

            scores = (images - reconstructions).square().mean(dim=(1, 2, 3))

            if not torch.isfinite(scores).all().item():
                raise ValueError("Wyniki zawierają NaN lub Inf.")

            paths.extend(batch["path"])
            score_values.extend(scores.cpu().tolist())

    scores = np.asarray(score_values, dtype=np.float64)
    if len(scores) != len(dataset):
        raise ValueError("Nie obliczono wyniku dla każdego zdjęcia.")

    quantile = 0.95
    threshold = float(np.quantile(scores, quantile, method="higher"))
    alarms = scores > threshold
    false_alarms = int(alarms.sum())
    observed_alarm_rate = float(alarms.mean())

    output_dir = run_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV pozwala sprawdzić wynik każdego zdjęcia.
    with (output_dir / "validation_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["path", "score", "predicted_anomaly"])
        for path, score, alarm in zip(paths, scores, alarms):
            writer.writerow([path, float(score), int(alarm)])

    # JSON zapisuje regułę, którą później zastosujemy do testu.
    calibration = {
        "schema_version": 1,
        "checkpoint": "best.pt",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "score": "mean_squared_reconstruction_error",
        "image_size": config["image_size"],
        "split": "validation",
        "num_images": len(scores),
        "quantile": quantile,
        "quantile_method": "higher",
        "threshold": threshold,
        "decision_rule": "score > threshold",
        "mean_validation_score": float(scores.mean()),
        "false_alarms_on_calibration": false_alarms,
        "observed_alarm_rate_on_calibration": observed_alarm_rate,
        "independent_of_checkpoint_selection": False,
    }
    (output_dir / "threshold.json").write_text(
        json.dumps(calibration, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Validation images: {len(scores)}")
    print(f"Mean validation score: {scores.mean():.10f}")
    print(f"Saved validation MSE: {checkpoint['validation_mse']:.10f}")
    print(f"Threshold (95th percentile, higher): {threshold:.10f}")
    print(f"False alarms on calibration: {false_alarms}/{len(scores)}")
    print(f"Observed calibration alarm rate: {observed_alarm_rate:.2%}")
    print(f"Output directory: {output_dir}")
    print("CALIBRATION: OK")


if __name__ == "__main__":
    main()
