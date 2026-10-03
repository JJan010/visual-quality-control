import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from evaluate_patchcore import BottleTestDataset, sha256_file


def load_mask(dataset_root, relative_path, category, size):
    if category == "good":
        return np.zeros((size, size), dtype=bool)

    stem = Path(relative_path).stem
    mask_path = (
        dataset_root / "ground_truth" / category / f"{stem}_mask.png"
    )

    with Image.open(mask_path) as file:
        mask_image = file.convert("L")

        # Oczekujemy binarnej maski zapisanej jako 0 i 255.
        values = np.unique(np.asarray(mask_image))
        if not np.isin(values, [0, 255]).all():
            raise RuntimeError(f"Nieoczekiwane wartości maski: {mask_path}")

        mask = np.asarray(
            mask_image.resize((size, size), Image.Resampling.NEAREST)
        ) > 0

    if not mask.any():
        raise RuntimeError(f"Pusta maska defektu po skalowaniu: {mask_path}")

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()
    checkpoint_path = run_dir / "model.pt"
    output_dir = run_dir / "localization"

    if output_dir.exists():
        raise SystemExit(
            "Katalog localization już istnieje. Zachowano poprzednie wyniki."
        )
    if not torch.cuda.is_available():
        raise SystemExit("STOP: GPU jest niedostępne.")

    image_metrics = json.loads(
        (run_dir / "evaluation/metrics.json").read_text(encoding="utf-8")
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != image_metrics["checkpoint_sha256"]:
        raise RuntimeError("Model różni się od użytego podczas ewaluacji.")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]

    for name, expected in checkpoint["versions"].items():
        if version(name) != expected:
            raise RuntimeError(f"Wersja {name} różni się od zapisanej.")

    size = config["image_size"]
    dataset = BottleTestDataset(
        root=project_root / checkpoint["split"]["dataset_root"],
        image_size=size,
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

    model = PatchcoreModel(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=False,
        num_neighbors=config["num_neighbors"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to("cuda")
    model.eval()
    model.requires_grad_(False)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    all_masks = []
    all_maps = []
    image_records = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = normalize(batch["image"].to("cuda"))
            prediction = model(images)

            # Każda mapa ma jeden kanał i rozmiar obrazu wejściowego.
            maps = prediction.anomaly_map
            expected_shape = (images.shape[0], 1, size, size)
            if tuple(maps.shape) != expected_shape:
                raise RuntimeError(f"Nieprawidłowy kształt map: {maps.shape}")
            if not torch.isfinite(maps).all().item():
                raise RuntimeError("Mapy zawierają NaN lub Inf.")

            maps = maps[:, 0].cpu().numpy()

            for path, category, anomaly_map in zip(
                batch["path"], batch["category"], maps
            ):
                mask = load_mask(dataset.root, path, category, size)

                # Te same pozycje w obu tablicach opisują te same piksele.
                all_masks.append(mask.reshape(-1))
                all_maps.append(anomaly_map.reshape(-1))

                image_records.append({
                    "path": path,
                    "category": category,
                    "defect_pixels": int(mask.sum()),
                })

            print(
                f"Localization: batch {batch_index}/{len(loader)}",
                flush=True,
            )

    if len(image_records) != len(dataset):
        raise RuntimeError("Nie przetworzono wszystkich zdjęć.")

    labels = np.concatenate(all_masks)
    scores = np.concatenate(all_maps)

    assert labels.size == len(dataset) * size * size
    assert labels.any() and (~labels).any()

    print("Computing pixel metrics on CPU...", flush=True)
    pixel_auroc = float(roc_auc_score(labels, scores))
    pixel_ap = float(average_precision_score(labels, scores))

    metrics = {
        "checkpoint_sha256": checkpoint_hash,
        "test_images": len(dataset),
        "evaluation_resolution": [size, size],
        "aggregation": "pooled pixels from all test images",
        "score": "raw anomalib anomaly_map",
        "per_image_map_normalization": False,
        "mask_resize": "PIL NEAREST",
        "pixel_threshold": None,
        "total_pixels": int(labels.size),
        "defect_pixels": int(labels.sum()),
        "normal_pixels": int((~labels).sum()),
        "defect_pixel_fraction": float(labels.mean()),
        "pixel_auroc": pixel_auroc,
        "pixel_average_precision": pixel_ap,
        "metric_library": {
            "scikit-learn": version("scikit-learn"),
            "numpy": version("numpy"),
        },
    }

    output_dir.mkdir(exist_ok=False)
    for name, content in (
        ("metrics.json", metrics),
        ("evaluated_images.json", image_records),
    ):
        (output_dir / name).write_text(
            json.dumps(content, indent=2) + "\n", encoding="utf-8"
        )

    print("Inference device: cuda")
    print("Test images:", len(dataset))
    print("Evaluation resolution:", (size, size))
    print("Total pixels:", labels.size)
    print("Defect pixels:", int(labels.sum()))
    print(f"Defect pixel fraction: {labels.mean():.2%}")
    print(f"Pixel AUROC: {pixel_auroc:.6f}")
    print(f"Pixel Average Precision: {pixel_ap:.6f}")
    print("Output directory:", output_dir)
    print("PATCHCORE LOCALIZATION: OK")


if __name__ == "__main__":
    main()
