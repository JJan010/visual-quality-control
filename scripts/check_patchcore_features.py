from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from visual_quality.data.bottle import BottleNormalDataset


def main():
    project_root = Path(__file__).resolve().parents[1]

    if not torch.cuda.is_available():
        raise SystemExit("STOP: GPU jest niedostępne.")

    device = torch.device("cuda")

    # Korzystamy z naszego istniejącego podziału danych.
    dataset = BottleNormalDataset(
        project_root=project_root,
        split="train",
        image_size=256,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    images = batch["image"].to(device)

    # Dataset zwraca RGB w zakresie 0–1.
    # Dodajemy normalizację kanałów stosowaną dla wag ImageNet.
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    print("Przygotowanie modelu; pierwszy start może pobrać wagi.")
    model = PatchcoreModel(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    ).to(device)

    model.eval()
    model.requires_grad_(False)

    with torch.inference_mode():
        normalized = normalize(images)

        # Odczytujemy mapy cech z dwóch poziomów sieci.
        features = model.feature_extractor(normalized)

        # Uśredniamy lokalne sąsiedztwo na każdej mapie.
        pooled = {
            name: model.feature_pooler(value)
            for name, value in features.items()
        }

        # Łączymy mapy na wspólnej siatce przestrzennej.
        embedding_map = model.generate_embedding(pooled)

        # Każdy wiersz opisuje jedną lokalizację jednego obrazu.
        patch_vectors = model.reshape_embedding(embedding_map)

    torch.cuda.synchronize()

    print("Pierwszy plik:", batch["path"][0])
    print("Wejście:", tuple(images.shape))

    for name, value in features.items():
        print(f"{name}:", tuple(value.shape))

    print("Połączona mapa:", tuple(embedding_map.shape))
    print("Wektory fragmentów:", tuple(patch_vectors.shape))
    print("Urządzenie:", patch_vectors.device)
    print("Typ:", patch_vectors.dtype)
    print("Śledzenie gradientów:", patch_vectors.requires_grad)

    assert tuple(embedding_map.shape) == (2, 1536, 32, 32)
    assert tuple(patch_vectors.shape) == (2048, 1536)
    assert patch_vectors.is_cuda
    assert not patch_vectors.requires_grad
    assert torch.isfinite(patch_vectors).all().item()

    print("PATCHCORE FEATURES: OK")


if __name__ == "__main__":
    main()
