import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

# Wykorzystujemy ten sam odczyt zdjęć i funkcję SHA-256 co w ewaluacji.
from evaluate_patchcore import BottleTestDataset, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()
    checkpoint_path = run_dir / "model.pt"
    output_dir = run_dir / "inspection"

    if not torch.cuda.is_available():
        raise SystemExit("STOP: GPU jest niedostępne.")

    calibration = json.loads(
        (run_dir / "calibration/threshold.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (run_dir / "evaluation/metrics.json").read_text(encoding="utf-8")
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    if not (
        checkpoint_hash
        == calibration["checkpoint_sha256"]
        == metrics["checkpoint_sha256"]
    ):
        raise RuntimeError("Model, kalibracja i ewaluacja nie są zgodne.")

    with (run_dir / "evaluation/test_scores.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    selected = []
    for category in ("broken_large", "broken_small", "contamination", "good"):
        candidates = [row for row in rows if row["category"] == category]
        choose = max if category == "good" else min
        selected.append(
            choose(candidates, key=lambda row: float(row["score"]))
        )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]

    for name, expected in checkpoint["versions"].items():
        if version(name) != expected:
            raise RuntimeError(f"Wersja {name} różni się od zapisanej.")

    dataset = BottleTestDataset(
        root=project_root / checkpoint["split"]["dataset_root"],
        image_size=config["image_size"],
    )
    index_by_path = {
        path.relative_to(dataset.root).as_posix(): index
        for index, path in enumerate(dataset.paths)
    }

    # Pobieramy wybrane zdjęcia dokładnie tak jak podczas ewaluacji.
    samples = [dataset[index_by_path[row["path"]]] for row in selected]
    images = torch.stack([sample["image"] for sample in samples])

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

    with torch.inference_mode():
        prediction = model(normalize(images.to("cuda")))
        scores = prediction.pred_score.reshape(-1).cpu().numpy()
        maps = prediction.anomaly_map[:, 0].cpu().numpy()

    size = config["image_size"]
    assert maps.shape == (4, size, size)
    assert np.isfinite(maps).all()
    assert np.isfinite(scores).all()

    # Inny rozmiar batcha może dawać niewielkie różnice numeryczne.
    saved_scores = np.array([float(row["score"]) for row in selected])
    if not np.allclose(scores, saved_scores, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Wyniki odbiegają od zapisanej ewaluacji.")

    masks = []
    for row in selected:
        if row["category"] == "good":
            mask = np.zeros((size, size), dtype=bool)
        else:
            stem = Path(row["path"]).stem
            mask_path = (
                dataset.root / "ground_truth" / row["category"]
                / f"{stem}_mask.png"
            )
            with Image.open(mask_path) as file:
                mask = np.asarray(
                    file.convert("L").resize(
                        (size, size), Image.Resampling.NEAREST
                    )
                ) > 0
            if not mask.any():
                raise RuntimeError(f"Pusta maska defektu: {mask_path}")
        masks.append(mask)

    # Jedna skala kolorów dla wszystkich czterech przykładów.
    vmax = max(float(maps.max()), 1e-8)
    fig, axes = plt.subplots(
        4, 4, figsize=(14, 14), layout="constrained"
    )
    titles = ("Input", "Ground truth", "Anomaly map", "Overlay + GT contour")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)

    for index, (row, mask) in enumerate(zip(selected, masks)):
        rgb = images[index].permute(1, 2, 0).numpy()

        axes[index, 0].imshow(rgb)
        axes[index, 0].set_ylabel(
            f"{row['category']}/{Path(row['path']).name}\n"
            f"Image score: {scores[index]:.3f}"
        )
        axes[index, 1].imshow(
            mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        heat = axes[index, 2].imshow(
            maps[index], cmap="magma", vmin=0, vmax=vmax
        )
        axes[index, 3].imshow(rgb)
        axes[index, 3].imshow(
            maps[index], cmap="magma", vmin=0, vmax=vmax, alpha=0.5
        )
        if mask.any():
            axes[index, 3].contour(
                mask.astype(float), levels=[0.5],
                colors="lime", linewidths=0.8,
            )

        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.colorbar(
        heat, ax=axes[:, 2:].ravel().tolist(),
        label="Raw anomaly-map value", shrink=0.7,
    )
    fig.suptitle(
        "PatchCore: lowest-scoring defect per category + highest-scoring good\n"
        f"Image-level threshold: {calibration['threshold']:.5f}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "localization_preview.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "selection_rule": "minimum score per defect category; maximum for good",
        "examples": selected,
        "map_color_limits": [0.0, vmax],
        "mask_resize": "nearest",
        "image_threshold": calibration["threshold"],
    }
    (output_dir / "selected_examples.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    for row, score in zip(selected, scores):
        print(f"{row['path']} | score: {score:.5f}")
    print("Saved preview:", image_path)
    print("PATCHCORE INSPECTION: OK")


if __name__ == "__main__":
    main()
