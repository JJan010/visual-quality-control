import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from visual_quality.data.bottle import BottleNormalDataset
from visual_quality.models.autoencoder import ConvAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    saved_split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    current_split = json.loads(
        (project_root / "configs" / "splits" / "bottle_v1.json")
        .read_text(encoding="utf-8")
    )

    if saved_split != current_split:
        raise ValueError("Current split differs from the training split.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    with (run_dir / "history.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        history = list(csv.DictReader(file))

    epochs = [int(row["epoch"]) for row in history]
    train_mse = [float(row["train_mse"]) for row in history]
    validation_mse = [float(row["validation_mse"]) for row in history]

    output_dir = run_dir / "inspection"
    output_dir.mkdir(exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    axis.plot(epochs, train_mse, label="Training MSE")
    axis.plot(epochs, validation_mse, label="Validation MSE")
    axis.scatter(
        checkpoint["epoch"],
        checkpoint["validation_mse"],
        color="black",
        label="Saved checkpoint",
        zorder=3,
    )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE (log scale)")
    axis.set_title("Autoencoder learning curves")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    dataset = BottleNormalDataset(
        project_root,
        split="validation",
        image_size=checkpoint["config"]["image_size"],
    )

    samples = [dataset[index] for index in range(4)]
    images = torch.stack([sample["image"] for sample in samples])
    images = images.to(device)

    with torch.inference_mode():
        reconstructions = model(images)

    if reconstructions.shape != images.shape:
        raise ValueError("Reconstruction shape does not match the input.")

    if not torch.isfinite(reconstructions).all().item():
        raise ValueError("Reconstruction contains non-finite values.")

    errors = (images - reconstructions).square().mean(dim=1).cpu()
    images = images.cpu()
    reconstructions = reconstructions.cpu()

    # Use one shared color scale for all four error maps.
    error_max = max(errors.max().item(), 1e-12)

    fig, axes = plt.subplots(
        4, 3, figsize=(11, 12), layout="constrained"
    )

    for index, sample in enumerate(samples):
        axes[index, 0].imshow(images[index].permute(1, 2, 0).numpy())
        axes[index, 0].set_title(sample["path"])

        axes[index, 1].imshow(
            reconstructions[index].permute(1, 2, 0).numpy()
        )
        axes[index, 1].set_title("Reconstruction")

        heatmap = axes[index, 2].imshow(
            errors[index].numpy(),
            cmap="inferno",
            vmin=0,
            vmax=error_max,
        )
        axes[index, 2].set_title(
            f"Error map | MSE={errors[index].mean().item():.5f}"
        )

        for axis in axes[index]:
            axis.axis("off")

    fig.colorbar(
        heatmap,
        ax=axes[:, 2].tolist(),
        shrink=0.75,
        label="Squared error averaged over RGB",
    )
    fig.savefig(output_dir / "validation_reconstructions.png", dpi=150)
    plt.close(fig)

    print("Loaded epoch:", checkpoint["epoch"])
    print("Saved validation MSE:", checkpoint["validation_mse"])
    print("Device:", device)
    print("Output directory:", output_dir)
    print("INSPECTION: OK")


if __name__ == "__main__":
    main()
