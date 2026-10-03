import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.patchcore import PatchcorePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--blur-backend",
        choices=("original_2d", "separable_1d"),
        default="original_2d",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    predictor = PatchcorePredictor(
        run_dir,
        device="cuda",
        blur_backend=args.blur_backend,
    )

    expected_class = {
        "original_2d": "GaussianBlur2d",
        "separable_1d": "SeparableGaussianBlur2d",
    }[args.blur_backend]
    actual_class = type(predictor.model.anomaly_map_generator.blur).__name__
    if actual_class != expected_class:
        raise RuntimeError(f"Wczytano niewłaściwy filtr: {actual_class}")

    metrics = json.loads(
        (run_dir / "evaluation/metrics.json").read_text(encoding="utf-8")
    )
    if metrics["checkpoint_sha256"] != predictor.checkpoint_sha256:
        raise RuntimeError("Wyniki referencyjne dotyczą innego modelu.")

    with (run_dir / "evaluation/test_scores.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        reference_rows = list(csv.DictReader(file))

    reference = {row["path"]: row for row in reference_rows}
    if len(reference) != len(reference_rows):
        raise RuntimeError("Powtórzone ścieżki w wynikach referencyjnych.")

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        root=project_root / split["dataset_root"],
        image_size=predictor.image_size,
    )

    dataset_paths = {
        path.relative_to(dataset.root).as_posix()
        for path in dataset.paths
    }
    if dataset_paths != set(reference):
        raise RuntimeError("Zestaw zdjęć różni się od referencji.")

    loader = DataLoader(
        dataset,
        batch_size=predictor.config["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    actual_scores = []
    expected_scores = []
    actual_labels = []
    expected_labels = []

    for batch in loader:
        prepared = []
        for relative_path in batch["path"]:
            with Image.open(dataset.root / relative_path) as image:
                prepared.append(predictor.prepare_image(image))
        images = torch.stack(prepared)

        # Porównujemy nowy preprocessing z dotychczasowym.
        if not torch.equal(images, batch["image"]):
            raise RuntimeError("Przygotowanie obrazu zmieniło wartości tensora.")

        result = predictor.predict_batch(images)

        expected_map_shape = (
            len(batch["path"]), 1,
            predictor.image_size, predictor.image_size,
        )
        assert tuple(result.anomaly_maps.shape) == expected_map_shape
        assert torch.isfinite(result.anomaly_maps).all().item()
        assert torch.isfinite(result.scores).all().item()

        actual_scores.extend(result.scores.cpu().tolist())
        actual_labels.extend(result.labels.cpu().tolist())

        for path in batch["path"]:
            expected_scores.append(float(reference[path]["score"]))
            expected_labels.append(bool(int(reference[path]["predicted_anomaly"])))

    actual_scores = np.asarray(actual_scores)
    expected_scores = np.asarray(expected_scores)
    label_changes = int(np.count_nonzero(
        np.asarray(actual_labels) != np.asarray(expected_labels)
    ))
    max_error = float(np.max(np.abs(actual_scores - expected_scores)))

    # Dopuszczamy małe różnice obliczeń zmiennoprzecinkowych.
    if not np.allclose(
        actual_scores, expected_scores, rtol=1e-4, atol=1e-4
    ):
        raise RuntimeError(f"Wyniki odbiegają od referencji: max error={max_error}")
    if label_changes:
        raise RuntimeError(f"Liczba zmienionych decyzji: {label_changes}")

    report = {
        "blur_backend": predictor.blur_backend,
        "blur_class": type(predictor.model.anomaly_map_generator.blur).__name__,
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "checked_images": len(actual_scores),
        "preprocessing_exact_match": True,
        "score_rtol": 1e-4,
        "score_atol": 1e-4,
        "max_absolute_score_difference": max_error,
        "changed_decisions": label_changes,
        "anomaly_maps_shape_and_finiteness": "passed",
    }
    output_dir = run_dir / "verification"
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"predictor_parity_{args.blur_backend}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("Blur backend:", predictor.blur_backend)
    print("Blur class:", type(predictor.model.anomaly_map_generator.blur).__name__)
    print("Checked images:", len(actual_scores))
    print("Preprocessing: exact match")
    print(f"Maximum score difference: {max_error:.10f}")
    print("Changed decisions:", label_changes)
    print("Anomaly maps: shape and finite values OK")
    print("PATCHCORE PREDICTOR: OK")


if __name__ == "__main__":
    main()
