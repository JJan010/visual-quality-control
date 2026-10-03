"""Kalibracja progu mapy na poprawnych obrazach walidacyjnych."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from visual_quality.inference.patchcore import file_sha256
from visual_quality.inference.runtime import load_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/runtime/patchcore_tensorrt.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    if args.output_dir and args.output_dir.exists():
        raise RuntimeError("Output directory already exists; calibration was preserved.")

    predictor = load_predictor(config_path)

    # Używamy podziału zapisanego przy budowie tego modelu.
    split_path = predictor.run_dir / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = (project_root / split["dataset_root"]).resolve()
    paths = split["validation"]
    train_paths = set(split["train"])

    if len(paths) != 42 or len(set(paths)) != 42:
        raise RuntimeError("Oczekiwano 42 różnych obrazów walidacyjnych.")
    if train_paths.intersection(paths):
        raise RuntimeError("Zbiory treningowy i walidacyjny nakładają się.")
    if not all(path.startswith("train/good/") for path in paths):
        raise RuntimeError("Kalibracja wymaga wyłącznie poprawnych obrazów.")

    records = []

    for index, relative_path in enumerate(paths, start=1):
        path = dataset_root / relative_path

        with Image.open(path) as image:
            tensor = predictor.prepare_image(image)

        prediction = predictor.predict_batch(tensor.unsqueeze(0))
        maps = prediction.anomaly_maps.detach().cpu().numpy()

        expected_shape = (1, 1, predictor.image_size, predictor.image_size)
        if maps.shape != expected_shape:
            raise RuntimeError(f"Nieoczekiwany kształt mapy: {maps.shape}")
        if not np.isfinite(maps).all():
            raise RuntimeError(f"Mapa zawiera NaN lub Inf: {relative_path}")

        records.append({
            "path": relative_path,
            "image_sha256": file_sha256(path),
            "map_max": float(maps[0, 0].max()),
        })

        print(f"Calibration image {index}/{len(paths)}")

    maxima = np.asarray(
        [record["map_max"] for record in records],
        dtype=np.float64,
    )

    quantile = 0.95
    threshold = float(np.quantile(maxima, quantile, method="higher"))
    alarms = int(np.count_nonzero(maxima > threshold))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (args.output_dir.resolve() if args.output_dir else
                  predictor.run_dir / "localization_calibration" / stamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "schema_version": 1,
        "purpose": "localization_threshold",
        "method": "quantile_of_normal_validation_map_maxima",
        "quantile": quantile,
        "quantile_method": "higher",
        "pixel_decision_rule": "anomaly_map > threshold",
        "threshold": threshold,
        "calibration_images": len(paths),
        "images_with_any_flagged_pixel": alarms,
        "observed_image_alarm_rate": alarms / len(paths),
        "image_size": predictor.image_size,
        "batch_size": 1,
        "split_sha256": file_sha256(split_path),
        "runtime_config_sha256": file_sha256(config_path),
        "runtime": predictor.runtime_metadata(),
    }

    (output_dir / "threshold.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "validation_map_maxima.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "image_sha256", "map_max"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)

    print()
    print("Backend:", predictor.backend)
    print("Blur backend:", predictor.blur_backend)
    print("Validation images:", len(paths))
    print(f"Localization threshold: {threshold:.8f}")
    print(f"Images with any flagged pixel: {alarms}/{len(paths)}")
    print(f"Observed calibration alarm rate: {alarms / len(paths):.2%}")
    print("Image classification threshold unchanged:", predictor.threshold)
    print("Output directory:", output_dir)
    print("LOCALIZATION CALIBRATION: OK")


if __name__ == "__main__":
    main()
