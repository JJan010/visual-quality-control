import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from visual_quality.data.bottle import BottleNormalDataset


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()
    checkpoint_path = run_dir / "model.pt"
    output_dir = run_dir / "calibration"

    if output_dir.exists():
        raise SystemExit(
            "Katalog calibration już istnieje. Zachowano poprzednie wyniki."
        )
    if not torch.cuda.is_available():
        raise SystemExit("STOP: GPU jest niedostępne.")

    # Wczytujemy nasz zapisany model razem z jego konfiguracją.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    saved_split = checkpoint["split"]

    current_split = json.loads(
        (project_root / "configs/splits/bottle_v1.json").read_text(
            encoding="utf-8"
        )
    )
    if current_split != saved_split:
        raise RuntimeError("Podział danych różni się od zapisanego w modelu.")

    for name, expected in checkpoint["versions"].items():
        actual = version(name)
        if actual != expected:
            raise RuntimeError(
                f"Inna wersja {name}: obecnie {actual}, zapisano {expected}."
            )

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Nie pobieramy wag: zastąpimy je zawartością checkpointu.
    model = PatchcoreModel(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=False,
        num_neighbors=config["num_neighbors"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    if model.memory_bank.shape[0] == 0:
        raise RuntimeError("Wczytany bank cech jest pusty.")

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
    normalize = v2.Normalize(
        mean=config["normalization_mean"],
        std=config["normalization_std"],
    )

    paths = []
    scores = []

    with torch.inference_mode():
        for batch in loader:
            images = normalize(batch["image"].to(device))

            # Pełna inferencja: cechy, porównanie z bankiem, wynik i mapa.
            prediction = model(images)
            batch_scores = prediction.pred_score.reshape(-1)

            if batch_scores.numel() != images.shape[0]:
                raise RuntimeError("Nieprawidłowa liczba wyników.")
            if not torch.isfinite(batch_scores).all().item():
                raise RuntimeError("Wyniki zawierają NaN lub Inf.")
            if not torch.isfinite(prediction.anomaly_map).all().item():
                raise RuntimeError("Mapa anomalii zawiera NaN lub Inf.")

            paths.extend(batch["path"])
            scores.extend(batch_scores.cpu().tolist())

    if paths != saved_split["validation"]:
        raise RuntimeError("Odczytane zdjęcia nie zgadzają się z manifestem.")

    scores = np.asarray(scores, dtype=np.float64)
    threshold = float(np.quantile(scores, 0.95, method="higher"))
    alarms = scores > threshold
    alarm_count = int(alarms.sum())

    metadata = {
        "model": "PatchCore",
        "checkpoint": "model.pt",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "score_definition": "anomalib.PatchcoreModel.pred_score",
        "threshold": threshold,
        "quantile": 0.95,
        "quantile_method": "higher",
        "decision_rule": "score > threshold",
        "calibration_split": "validation",
        "calibration_images": len(scores),
        "calibration_alarms": alarm_count,
        "config": config,
        "versions": checkpoint["versions"],
    }

    output_dir.mkdir(exist_ok=False)
    (output_dir / "threshold.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "validation_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "score", "predicted_anomaly"],
        )
        writer.writeheader()
        for path, score, alarm in zip(paths, scores, alarms):
            writer.writerow({
                "path": path,
                "score": float(score),
                "predicted_anomaly": int(alarm),
            })

    print("Device:", device)
    print("Loaded memory bank:", tuple(model.memory_bank.shape))
    print("Validation images:", len(scores))
    print(f"Minimum score: {scores.min():.8f}")
    print(f"Mean score: {scores.mean():.8f}")
    print(f"Maximum score: {scores.max():.8f}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Calibration alarms: {alarm_count}/{len(scores)}")
    print(f"Calibration alarm rate: {alarms.mean():.2%}")
    print("Output directory:", output_dir)
    print("PATCHCORE CALIBRATION: OK")


if __name__ == "__main__":
    main()
