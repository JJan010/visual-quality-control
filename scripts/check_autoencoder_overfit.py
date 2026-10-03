import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn

from visual_quality.data.bottle import BottleNormalDataset
from visual_quality.models.autoencoder import ConvAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps musi być dodatnie.")

    project_root = Path(__file__).resolve().parents[1]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA niedostępna.")

    config = {
        "seed": 42,
        "num_images": 4,
        "image_size": 256,
        "steps": args.steps,
        "learning_rate": 0.001,
        "purpose": "diagnostic overfit on a fixed training batch",
    }

    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    dataset = BottleNormalDataset(
        project_root=project_root,
        split="train",
        image_size=config["image_size"],
    )
    samples = [dataset[i] for i in range(config["num_images"])]
    paths = [sample["path"] for sample in samples]
    images = torch.stack([sample["image"] for sample in samples]).to(device)

    # Nowy model. Nie wczytujemy ani nie nadpisujemy best.pt.
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = project_root / "artifacts/diagnostics" / f"ae_overfit_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    config["paths"] = paths
    config["torch_version"] = str(torch.__version__)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    history = []

    def measure(step):
        model.eval()
        with torch.inference_mode():
            mse = criterion(model(images), images).item()
        history.append({"step": step, "mse": mse})
        print(f"Step {step:04d} | fixed-batch MSE: {mse:.8f}")
        model.train()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training images: {len(samples)}")
    print(f"Output directory: {output_dir}")
    measure(0)

    for step in range(1, config["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(images)
        loss = criterion(reconstruction, images)

        if not torch.isfinite(loss).item():
            raise ValueError("Loss zawiera NaN lub Inf.")

        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == config["steps"]:
            measure(step)

    model.eval()
    with torch.inference_mode():
        reconstructions = model(images).cpu()
    inputs = images.cpu()

    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["step", "mse"])
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(
        len(samples), 2, figsize=(8, 12), layout="constrained"
    )
    for index, path in enumerate(paths):
        axes[index, 0].imshow(inputs[index].permute(1, 2, 0).numpy())
        axes[index, 0].set_title(path)

        axes[index, 1].imshow(
            reconstructions[index].permute(1, 2, 0).numpy()
        )
        axes[index, 1].set_title(
            f"Reconstruction after {config['steps']} steps"
        )

        for axis in axes[index]:
            axis.axis("off")

    fig.suptitle("Diagnostic: four training images, repeatedly fitted")
    fig.savefig(output_dir / "reconstructions.png", dpi=140)
    plt.close(fig)

    print(f"Initial MSE: {history[0]['mse']:.8f}")
    print(f"Final MSE: {history[-1]['mse']:.8f}")
    print("DIAGNOSTIC COMPLETED")


if __name__ == "__main__":
    main()
