import json
import random
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from anomalib.models.components.sampling import KCenterGreedy
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from visual_quality.data.bottle import BottleNormalDataset


def main():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs/patchcore_baseline.json"
    split_path = project_root / "configs/splits/bottle_v1.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))

    if not torch.cuda.is_available():
        raise SystemExit("STOP: GPU jest niedostępne.")

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = project_root / "artifacts/runs" / f"patchcore_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Zachowujemy konfigurację i podział użyte w tym uruchomieniu.
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "split.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )

    dataset = BottleNormalDataset(
        project_root=project_root,
        split="train",
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

    model = PatchcoreModel(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=True,
        num_neighbors=config["num_neighbors"],
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Training images:", len(dataset), flush=True)
    print("Run directory:", run_dir, flush=True)

    # Etap 1: obliczamy cechy na GPU, zbieramy je w pamięci RAM.
    chunks = []
    image_paths = []

    torch.cuda.synchronize()
    extraction_start = perf_counter()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = normalize(batch["image"].to(device))

            features = model.feature_extractor(images)
            pooled = {
                name: model.feature_pooler(value)
                for name, value in features.items()
            }
            embedding_map = model.generate_embedding(pooled)
            vectors = model.reshape_embedding(embedding_map)

            if not torch.isfinite(vectors).all().item():
                raise RuntimeError("Cechy zawierają NaN lub Inf.")

            chunks.append(vectors.cpu())
            image_paths.extend(batch["path"])

            print(
                f"Features: batch {batch_index}/{len(loader)}",
                flush=True,
            )

    torch.cuda.synchronize()
    extraction_seconds = perf_counter() - extraction_start

    if image_paths != split["train"]:
        raise RuntimeError("Odczytane zdjęcia nie zgadzają się z manifestem.")

    all_embeddings = torch.cat(chunks, dim=0).to(device)
    del chunks

    print("All embeddings:", tuple(all_embeddings.shape), flush=True)

    # Etap 2: wybieramy reprezentatywny podzbiór na GPU.
    print("Selecting coreset...", flush=True)
    torch.cuda.synchronize()
    coreset_start = perf_counter()

    with torch.inference_mode():
        sampler = KCenterGreedy(
            embedding=all_embeddings,
            sampling_ratio=config["coreset_sampling_ratio"],
        )
        memory_bank = sampler.sample_coreset()

    torch.cuda.synchronize()
    coreset_seconds = perf_counter() - coreset_start

    expected_size = int(
        all_embeddings.shape[0] * config["coreset_sampling_ratio"]
    )
    assert memory_bank.shape == (expected_size, all_embeddings.shape[1])
    assert torch.isfinite(memory_bank).all().item()
    assert memory_bank.is_cuda

    model.memory_bank = memory_bank

    # Etap 3: zapisujemy wagi ekstraktora oraz bank cech.
    versions = {
        name: version(name)
        for name in ("torch", "torchvision", "anomalib", "timm")
    }
    checkpoint_path = run_dir / "model.pt"

    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "config": config,
            "split": split,
            "versions": versions,
        },
        checkpoint_path,
    )

    bank_mib = memory_bank.numel() * memory_bank.element_size() / 1024**2
    summary = {
        "training_images": len(dataset),
        "all_embeddings_shape": list(all_embeddings.shape),
        "memory_bank_shape": list(memory_bank.shape),
        "memory_bank_mib": bank_mib,
        "extraction_seconds": extraction_seconds,
        "coreset_seconds": coreset_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "versions": versions,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("Memory bank:", tuple(memory_bank.shape))
    print(f"Memory bank size: {bank_mib:.2f} MiB")
    print(f"Feature extraction: {extraction_seconds:.2f} s")
    print(f"Coreset selection: {coreset_seconds:.2f} s")
    print("Saved checkpoint:", checkpoint_path)
    print("PATCHCORE BANK: OK")


if __name__ == "__main__":
    main()
