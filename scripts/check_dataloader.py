from pathlib import Path

import torch
from torch.utils.data import DataLoader

from visual_quality.data.bottle import BottleNormalDataset


def main():
    project_root = Path(__file__).resolve().parents[1]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    for split in ("train", "validation"):
        dataset = BottleNormalDataset(
            project_root=project_root,
            split=split,
            image_size=256,
        )

        generator = torch.Generator().manual_seed(42)

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=(split == "train"),
            num_workers=0,
            drop_last=False,
            generator=generator,
        )

        batch = next(iter(loader))
        images = batch["image"]

        assert images.shape == (8, 3, 256, 256)
        assert images.dtype == torch.float32
        assert len(batch["path"]) == 8

        print(f"\nSplit: {split}")
        print("Images in dataset:", len(dataset))
        print("Batches per pass:", len(loader))
        print("Batch shape:", tuple(images.shape))
        print("Before transfer:", images.device)
        print("First file:", batch["path"][0])

        images = images.to("cuda")

        assert torch.isfinite(images).all().item()
        assert images.device.type == "cuda"

        print("After transfer:", images.device)

    print("\nDATALOADER: OK")


if __name__ == "__main__":
    main()
