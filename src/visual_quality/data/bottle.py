import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class BottleNormalDataset(Dataset):
    """Load normal bottle images from a recorded train/validation split."""

    def __init__(
        self,
        project_root: str | Path,
        split: str,
        image_size: int = 256,
    ):
        super().__init__()

        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")

        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        project_root = Path(project_root).resolve()
        manifest_path = (
            project_root / "configs" / "splits" / "bottle_v1.json"
        )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        self.dataset_root = project_root / manifest["dataset_root"]
        self.relative_paths = manifest[split]

        if not self.relative_paths:
            raise ValueError(f"The '{split}' split is empty.")

        for relative_path in self.relative_paths:
            path = self.dataset_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Missing image: {path}")

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.relative_paths)

    def __getitem__(self, index):
        relative_path = self.relative_paths[index]
        image_path = self.dataset_root / relative_path

        with Image.open(image_path) as file:
            image = file.convert("RGB")
            tensor = self.transform(image)

        return {
            "image": tensor,
            "path": relative_path,
        }
