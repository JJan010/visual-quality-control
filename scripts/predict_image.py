"""Analiza pojedynczego obrazu przy użyciu skonfigurowanego PatchCore."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visual_quality.inference.runtime import load_predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wykryj anomalię na pojedynczym obrazie."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/predictions"),
    )
    parser.add_argument(
        "--save-map",
        action="store_true",
        help="Zapisz surową mapę anomalii jako plik NumPy.",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    image_path = args.image.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

    predictor = load_predictor(config_path)

    with Image.open(image_path) as image:
        original_width, original_height = image.size
        image_tensor = predictor.prepare_image(image)

    # prepare_image zwraca (C, H, W).
    # Model oczekuje batcha, więc dodajemy wymiar: (1, C, H, W).
    batch = image_tensor.unsqueeze(0)
    prediction = predictor.predict_batch(batch)

    # Dopiero na granicy aplikacji przenosimy wyniki na CPU.
    scores = prediction.scores.detach().cpu()
    labels = prediction.labels.detach().cpu()
    maps = prediction.anomaly_maps.detach().cpu()

    expected_map_shape = (
        1, 1, predictor.image_size, predictor.image_size
    )
    if tuple(scores.shape) != (1,) or tuple(labels.shape) != (1,):
        raise RuntimeError("Nieoczekiwany kształt wyniku klasyfikacji.")
    if tuple(maps.shape) != expected_map_shape:
        raise RuntimeError(f"Nieoczekiwany kształt mapy: {maps.shape}")
    if not torch.isfinite(scores).all().item():
        raise RuntimeError("Wynik anomalii zawiera NaN lub Inf.")
    if not torch.isfinite(maps).all().item():
        raise RuntimeError("Mapa anomalii zawiera NaN lub Inf.")

    score = float(scores[0].item())
    is_anomaly = bool(labels[0].item())

    timestamp = datetime.now(timezone.utc)
    output_dir = (
        args.output_root.expanduser().resolve()
        / timestamp.strftime("prediction_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    map_filename = None
    if args.save_map:
        map_filename = "anomaly_map.npy"
        np.save(
            output_dir / map_filename,
            maps[0, 0].numpy(),
            allow_pickle=False,
        )

    result = {
        "schema_version": 1,
        "created_at_utc": timestamp.isoformat(),
        "input": {
            "path": str(image_path),
            "original_width": original_width,
            "original_height": original_height,
        },
        "prediction": {
            "score": score,
            "threshold": predictor.threshold,
            "decision_rule": "score > threshold",
            "is_anomaly": is_anomaly,
        },
        "anomaly_map": {
            "file": map_filename,
            "height": int(maps.shape[-2]),
            "width": int(maps.shape[-1]),
            "coordinates": "resized_model_input",
            "values": "raw_anomaly_scores",
        },
        "config_path": str(config_path),
        "runtime": predictor.runtime_metadata(),
    }

    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    print(f"Image: {image_path.name}")
    print(f"Backend: {predictor.backend}")
    print(f"Blur backend: {predictor.blur_backend}")
    print(f"Score: {score:.8f}")
    print(f"Threshold: {predictor.threshold:.8f}")
    print(f"Decision: {'ANOMALY' if is_anomaly else 'NORMAL'}")
    print(f"Result: {result_path}")
    if map_filename is not None:
        print(f"Anomaly map: {output_dir / map_filename}")
    print("SINGLE IMAGE INFERENCE: OK")


if __name__ == "__main__":
    main()
