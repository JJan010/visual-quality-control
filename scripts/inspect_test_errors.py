import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from visual_quality.models.autoencoder import ConvAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (run_dir / "evaluation/metrics.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["epoch"] != metrics["checkpoint_epoch"]:
        raise ValueError("Model i wyniki testu pochodzą z różnych epok.")

    with (run_dir / "evaluation/test_scores.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    for row in records:
        row["score"] = float(row["score"])
        row["is_anomaly"] = int(row["is_anomaly"])
        row["predicted_anomaly"] = int(row["predicted_anomaly"])

    # Nazwa, prawdziwa etykieta, decyzja modelu, wybór najwyższego wyniku.
    rules = [
        ("FN", 1, 0, False),
        ("FP", 0, 1, True),
        ("TP", 1, 1, True),
        ("TN", 0, 0, False),
    ]

    selected = []
    for name, label, prediction, highest in rules:
        candidates = [
            row for row in records
            if row["is_anomaly"] == label
            and row["predicted_anomaly"] == prediction
        ]
        if not candidates:
            print(f"Skipping {name}: no matching images.")
            continue

        candidates.sort(key=lambda row: (row["score"], row["path"]))
        chosen = candidates[-1] if highest else candidates[0]
        selected.append({"case": name, **chosen})

    if not selected:
        raise ValueError("Brak przykładów do wyświetlenia.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA niedostępna.")

    device = torch.device("cuda")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    size = checkpoint["config"]["image_size"]
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset_root = project_root / split["dataset_root"]
    tensors = []
    masks = []

    for row in selected:
        image_path = dataset_root / row["path"]

        with Image.open(image_path) as image:
            original_size = image.size
            tensors.append(transform(image.convert("RGB")))

        if row["is_anomaly"]:
            mask_path = (
                dataset_root
                / "ground_truth"
                / row["category"]
                / f"{image_path.stem}_mask.png"
            )
            with Image.open(mask_path) as mask:
                if mask.size != original_size:
                    raise ValueError(f"Niezgodny rozmiar maski: {mask_path}")
                mask = mask.convert("L").resize(
                    (size, size),
                    resample=Image.Resampling.NEAREST,
                )
                masks.append(np.asarray(mask) > 0)
        else:
            masks.append(np.zeros((size, size), dtype=bool))

    images = torch.stack(tensors).to(device)

    with torch.inference_mode():
        reconstructions = model(images)
        if reconstructions.shape != images.shape:
            raise ValueError("Niezgodny kształt rekonstrukcji.")
        error_maps = (images - reconstructions).square().mean(dim=1)

    if not torch.isfinite(error_maps).all().item():
        raise ValueError("Mapa błędu zawiera NaN lub Inf.")

    images = images.cpu()
    reconstructions = reconstructions.cpu()
    error_maps = error_maps.cpu()

    # Kontrola zgodności z wcześniejszą oceną.
    recalculated = error_maps.mean(dim=(1, 2)).numpy()
    saved_scores = np.array([row["score"] for row in selected])
    if not np.allclose(recalculated, saved_scores, rtol=1e-3, atol=1e-7):
        raise ValueError(
            "Wyniki różnią się od CSV. Sprawdź model i przetwarzanie danych."
        )

    # Jeden zakres kolorów dla wszystkich czterech przykładów.
    vmax = max(float(error_maps.max()), 1e-12)
    output_dir = run_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(selected):
        fig, axes = plt.subplots(
            1, 4, figsize=(16, 4.8), layout="constrained"
        )

        axes[0].imshow(images[index].permute(1, 2, 0).numpy())
        axes[0].set_title("Input")

        axes[1].imshow(
            reconstructions[index].permute(1, 2, 0).numpy()
        )
        axes[1].set_title("Reconstruction")

        heatmap = axes[2].imshow(
            error_maps[index].numpy(),
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        axes[2].set_title("Reconstruction error")

        axes[3].imshow(
            masks[index],
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[3].set_title("Ground truth: white = defect")

        for axis in axes:
            axis.axis("off")

        fig.colorbar(
            heatmap,
            ax=axes[2],
            shrink=0.75,
            label="Squared error averaged over RGB",
        )
        fig.suptitle(
            f"{row['case']} | {row['path']}\n"
            f"Saved score: {row['score']:.7f} | "
            f"Threshold: {metrics['threshold']:.7f}"
        )

        output_path = output_dir / f"{row['case'].lower()}.png"
        fig.savefig(output_path, dpi=140)
        plt.close(fig)

        print(f"{row['case']}: {row['path']} -> {output_path.name}")

    report = {
        "selection": "extreme scores within FN, FP, TP and TN groups",
        "representative_sample": False,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold": metrics["threshold"],
        "error_map_vmin": 0.0,
        "error_map_vmax": vmax,
        "examples": selected,
    }
    (output_dir / "selected_examples.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print("ERROR ANALYSIS: OK")


if __name__ == "__main__":
    main()
