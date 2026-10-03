import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from visual_quality.data.bottle import BottleNormalDataset
from visual_quality.models.autoencoder import ConvAutoencoder


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_images = 0

    for batch in loader:
        images = batch["image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(images)
        loss = criterion(reconstruction, images)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_images += batch_size

    return total_loss / total_images


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_images = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            reconstruction = model(images)
            loss = criterion(reconstruction, images)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite validation loss.")

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_images += batch_size

    return total_loss / total_images


import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="autoencoder_baseline.json",
        help="Nazwa pliku konfiguracji w katalogu configs.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset = BottleNormalDataset(
        project_root, "train", config["image_size"]
    )
    validation_dataset = BottleNormalDataset(
        project_root, "validation", config["image_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = project_root / "artifacts" / "runs" / f"ae_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    split_path = project_root / "configs" / "splits" / "bottle_v1.json"
    (output_dir / "split.json").write_bytes(split_path.read_bytes())

    best_validation_mse = float("inf")
    best_epoch = None

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Run directory:", output_dir, flush=True)

    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_mse", "validation_mse"],
        )
        writer.writeheader()

        for epoch in range(1, config["epochs"] + 1):
            train_mse = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            validation_mse = validate(
                model, validation_loader, criterion, device
            )

            writer.writerow({
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
            })
            file.flush()

            improved = validation_mse < best_validation_mse

            if improved:
                best_validation_mse = validation_mse
                best_epoch = epoch

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "validation_mse": validation_mse,
                        "config": config,
                        "torch_version": str(torch.__version__),
                    },
                    output_dir / "best.pt",
                )

            marker = " | saved best" if improved else ""
            print(
                f"Epoch {epoch:02d}/{config['epochs']} | "
                f"train MSE: {train_mse:.6f} | "
                f"validation MSE: {validation_mse:.6f}"
                f"{marker}",
                flush=True,
            )

    print("Best epoch:", best_epoch)
    print("Best validation MSE:", best_validation_mse)
    print("Saved checkpoint:", output_dir / "best.pt")


if __name__ == "__main__":
    main()
