import argparse
import csv
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BottleTestDataset(Dataset):
    def __init__(self, root, image_size):
        self.root = root
        self.paths = sorted((root / "test").glob("*/*.png"))
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

        expected = {
            "broken_large": 20,
            "broken_small": 22,
            "contamination": 21,
            "good": 20,
        }
        actual = Counter(path.parent.name for path in self.paths)
        if dict(actual) != expected:
            raise RuntimeError(f"Nieoczekiwany skład testu: {dict(actual)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as file:
            image = self.transform(file.convert("RGB"))
        return {
            "image": image,
            "path": path.relative_to(self.root).as_posix(),
            "category": path.parent.name,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()
    checkpoint_path = run_dir / "model.pt"
    output_dir = run_dir / "evaluation"

    if output_dir.exists():
        raise SystemExit(
            "Katalog evaluation już istnieje. Zachowano poprzednie wyniki."
        )
    if not torch.cuda.is_available():
        raise SystemExit("STOP: GPU jest niedostępne.")

    calibration = json.loads(
        (run_dir / "calibration/threshold.json").read_text(encoding="utf-8")
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != calibration["checkpoint_sha256"]:
        raise RuntimeError("Próg został wyznaczony dla innego checkpointu.")
    if calibration["decision_rule"] != "score > threshold":
        raise RuntimeError("Nieobsługiwana reguła decyzji.")

    threshold = float(calibration["threshold"])
    if not np.isfinite(threshold):
        raise RuntimeError("Próg nie jest skończoną liczbą.")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    if config != calibration["config"]:
        raise RuntimeError("Konfiguracja modelu i kalibracji jest różna.")

    for name, expected in checkpoint["versions"].items():
        actual = version(name)
        if actual != expected:
            raise RuntimeError(
                f"Inna wersja {name}: obecnie {actual}, zapisano {expected}."
            )

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

    dataset = BottleTestDataset(
        root=project_root / checkpoint["split"]["dataset_root"],
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

    rows = []
    with torch.inference_mode():
        for batch in loader:
            images = normalize(batch["image"].to(device))
            prediction = model(images)
            scores = prediction.pred_score.reshape(-1)

            if scores.numel() != images.shape[0]:
                raise RuntimeError("Nieprawidłowa liczba wyników.")
            if not torch.isfinite(scores).all().item():
                raise RuntimeError("Wyniki zawierają NaN lub Inf.")

            for path, category, score in zip(
                batch["path"], batch["category"], scores.cpu().tolist()
            ):
                rows.append({
                    "path": path,
                    "category": category,
                    "is_anomaly": int(category != "good"),
                    "score": float(score),
                    "predicted_anomaly": int(score > threshold),
                })

    assert len(rows) == len(dataset)

    labels = np.array([row["is_anomaly"] for row in rows], dtype=bool)
    predictions = np.array(
        [row["predicted_anomaly"] for row in rows], dtype=bool
    )
    scores = np.array([row["score"] for row in rows], dtype=np.float64)

    tp = int((labels & predictions).sum())
    fn = int((labels & ~predictions).sum())
    fp = int((~labels & predictions).sum())
    tn = int((~labels & ~predictions).sum())

    # Przy braku alarmów przyjmujemy precision i F1 równe zero.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    per_category = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        per_category[category] = {
            "images": len(selected),
            "alarms": sum(row["predicted_anomaly"] for row in selected),
        }

    metrics = {
        "test_images": len(rows),
        "threshold": threshold,
        "decision_rule": "score > threshold",
        "checkpoint_sha256": checkpoint_hash,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": fpr,
        "f1": f1,
        "image_auroc": float(roc_auc_score(labels, scores)),
        "per_category": per_category,
    }

    output_dir.mkdir(exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "test_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Device:", device)
    print("Test images:", len(rows))
    print(f"Fixed threshold: {threshold:.8f}")
    print(f"TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
    for name in ("recall", "precision", "false_positive_rate", "f1"):
        print(f"{name}: {metrics[name]:.2%}")
    print(f"Image AUROC: {metrics['image_auroc']:.6f}")

    for category, result in per_category.items():
        print(
            f"{category}: alarms {result['alarms']}/{result['images']}"
        )

    print("Output directory:", output_dir)
    print("PATCHCORE EVALUATION: OK")


if __name__ == "__main__":
    main()
