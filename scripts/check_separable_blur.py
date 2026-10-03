import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.filters import SeparableGaussianBlur2d
from visual_quality.inference.patchcore import PatchcorePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = (project_root / args.run_dir).resolve()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    predictor = PatchcorePredictor(run_dir, device="cuda")
    generator = predictor.model.anomaly_map_generator
    original_blur = generator.blur

    if (
        original_blur.border_type != "reflect"
        or original_blur.padding != "same"
        or original_blur.channels != 1
    ):
        raise RuntimeError("Nieobsługiwana konfiguracja oryginalnego filtra.")

    optimized_blur = SeparableGaussianBlur2d(
        original_blur.kernel
    ).to(predictor.device)
    optimized_blur.eval()

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        root=project_root / split["dataset_root"],
        image_size=predictor.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=predictor.config["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Tolerancje ustalone przed obejrzeniem wyników.
    map_rtol = 1e-5
    map_atol = 1e-4

    checked_images = 0
    checked_pixels = 0
    changed_decisions = 0
    max_map_error = 0.0
    sum_map_error = 0.0
    max_score_error = 0.0

    try:
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"]

            generator.blur = original_blur
            reference = predictor.predict_batch(images)

            generator.blur = optimized_blur
            candidate = predictor.predict_batch(images)

            if not torch.isfinite(reference.anomaly_maps).all().item():
                raise RuntimeError("Oryginalna mapa zawiera NaN lub Inf.")
            if not torch.isfinite(candidate.anomaly_maps).all().item():
                raise RuntimeError("Nowa mapa zawiera NaN lub Inf.")

            map_difference = (
                candidate.anomaly_maps - reference.anomaly_maps
            ).abs()
            score_difference = (
                candidate.scores - reference.scores
            ).abs()

            max_map_error = max(
                max_map_error, map_difference.max().item()
            )
            sum_map_error += map_difference.double().sum().item()
            max_score_error = max(
                max_score_error, score_difference.max().item()
            )
            changed_decisions += int(
                (candidate.labels != reference.labels).sum().item()
            )
            checked_images += images.shape[0]
            checked_pixels += map_difference.numel()

            # Wynik zdjęcia nie powinien zależeć od wygładzania mapy.
            torch.testing.assert_close(
                candidate.scores,
                reference.scores,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                candidate.anomaly_maps,
                reference.anomaly_maps,
                rtol=map_rtol,
                atol=map_atol,
            )

            print(
                f"Compared batch {batch_index}/{len(loader)}",
                flush=True,
            )
    finally:
        generator.blur = original_blur

    assert checked_images == len(dataset)
    assert changed_decisions == 0

    report = {
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "checked_images": checked_images,
        "checked_pixels": checked_pixels,
        "reference_kernel_shape": list(original_blur.kernel.shape),
        "padding": "same",
        "border_type": "reflect",
        "map_rtol": map_rtol,
        "map_atol": map_atol,
        "max_absolute_map_difference": max_map_error,
        "mean_absolute_map_difference": sum_map_error / checked_pixels,
        "max_absolute_score_difference": max_score_error,
        "changed_decisions": changed_decisions,
        "status": "passed",
    }

    output_dir = run_dir / "verification"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "separable_blur_parity.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Checked images:", checked_images)
    print("Compared map pixels:", checked_pixels)
    print(f"Maximum map difference: {max_map_error:.10f}")
    print(
        f"Mean map difference: "
        f"{sum_map_error / checked_pixels:.10f}"
    )
    print(f"Maximum score difference: {max_score_error:.10f}")
    print("Changed decisions:", changed_decisions)
    print("SEPARABLE BLUR CHECK: OK")


if __name__ == "__main__":
    main()
