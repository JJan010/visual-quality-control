import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from torch import nn
from torch.utils.data import DataLoader

from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.patchcore import PatchcorePredictor


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DiagnosticOnnxExtractor(nn.Module):
    """Correctness adapter with CPU transfers; not for benchmarking."""

    def __init__(self, session):
        super().__init__()
        self.session = session

    def forward(self, images):
        inputs = images.detach().cpu().contiguous().numpy()
        outputs = self.session.run(
            ["layer2", "layer3"],
            {"images": inputs},
        )
        return {
            name: torch.from_numpy(array).to(images.device)
            for name, array in zip(("layer2", "layer3"), outputs)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("baseline", "ieee"),
        default="baseline",
        help="Keep baseline settings or force IEEE FP32 in PyTorch.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    export_dir = args.export_dir.resolve()
    onnx_path = export_dir / "feature_extractor.onnx"

    diagnostic = json.loads(
        (export_dir / "numerical_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint_sha = sha256_file(run_dir / "model.pt")
    onnx_sha = sha256_file(onnx_path)

    if diagnostic["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("STOP: checkpoint różni się od diagnostyki.")
    if diagnostic["onnx_sha256"] != onnx_sha:
        raise RuntimeError("STOP: model ONNX różni się od diagnostyki.")

    if args.precision == "ieee":
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"

    predictor = PatchcorePredictor(
        run_dir,
        device="cuda",
        blur_backend="original_2d",
    )
    original_extractor = predictor.model.feature_extractor

    ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
    session = ort.InferenceSession(
        str(onnx_path),
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})
        ],
    )
    session.disable_fallback()
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("STOP: brak CUDA w sesji ONNX Runtime.")

    onnx_extractor = DiagnosticOnnxExtractor(session).eval()

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        project_root / split["dataset_root"],
        predictor.image_size,
    )
    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )

    with (run_dir / "evaluation/test_scores.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        saved_rows = list(csv.DictReader(handle))

    saved_scores = {
        row["path"]: float(row["score"]) for row in saved_rows
    }
    if len(saved_scores) != len(saved_rows):
        raise RuntimeError("STOP: powtórzone ścieżki w wynikach bazowych.")

    rows = []
    map_max = 0.0
    map_abs_sum = 0.0
    map_squared_error = 0.0
    map_squared_reference = 0.0
    map_pixels = 0
    map_outside = 0

    # Preserve the previous tolerance as a diagnostic measurement.
    atol = 1e-4
    rtol = 1e-4

    try:
        for batch_index, batch in enumerate(loader, start=1):
            predictor.model.feature_extractor = original_extractor
            reference = predictor.predict_batch(batch["image"])

            predictor.model.feature_extractor = onnx_extractor
            candidate = predictor.predict_batch(batch["image"])

            ref_scores = reference.scores.detach().cpu().numpy()
            new_scores = candidate.scores.detach().cpu().numpy()
            ref_maps = reference.anomaly_maps.detach().cpu().numpy()
            new_maps = candidate.anomaly_maps.detach().cpu().numpy()

            expected_shape = (
                len(batch["path"]), 1,
                predictor.image_size, predictor.image_size,
            )
            if ref_maps.shape != expected_shape or new_maps.shape != expected_shape:
                raise RuntimeError("STOP: niepoprawny kształt map.")

            for array in (ref_scores, new_scores, ref_maps, new_maps):
                if not np.isfinite(array).all():
                    raise RuntimeError("STOP: NaN lub Inf w wynikach.")

            # First confirm that the PyTorch reference reproduces our baseline.
            expected_scores = np.array([
                saved_scores[path] for path in batch["path"]
            ])
            if args.precision == "baseline":
                np.testing.assert_allclose(
                    ref_scores, expected_scores, atol=atol, rtol=rtol,
                    err_msg="PyTorch nie odtwarza zapisanych wyników bazowych.",
                )
            # IEEE mode may differ from the saved TF32-enabled baseline.
            # Those differences are recorded in the report below.

            ref64 = ref_maps.astype(np.float64)
            new64 = new_maps.astype(np.float64)
            difference = np.abs(new64 - ref64)

            map_max = max(map_max, float(difference.max()))
            map_abs_sum += float(difference.sum())
            map_squared_error += float(np.square(difference).sum())
            map_squared_reference += float(np.square(ref64).sum())
            map_pixels += difference.size
            map_outside += int(np.count_nonzero(
                difference > atol + rtol * np.abs(ref64)
            ))

            for path, ref_score, new_score in zip(
                batch["path"], ref_scores, new_scores
            ):
                ref_score = float(ref_score)
                new_score = float(new_score)
                rows.append({
                    "path": path,
                    "torch_score": ref_score,
                    "onnx_score": new_score,
                    "absolute_difference": abs(new_score - ref_score),
                    "torch_alarm": ref_score > predictor.threshold,
                    "onnx_alarm": new_score > predictor.threshold,
                })

            print(f"Compared batch {batch_index}/{len(loader)}", flush=True)
    finally:
        predictor.model.feature_extractor = original_extractor

    checked_paths = [row["path"] for row in rows]
    if len(set(checked_paths)) != len(rows):
        raise RuntimeError("STOP: powtórzone obrazy w porównaniu.")
    if set(checked_paths) != set(saved_scores):
        raise RuntimeError("STOP: inny zestaw obrazów niż w baseline.")

    changed_decisions = sum(
        row["torch_alarm"] != row["onnx_alarm"] for row in rows
    )
    score_outside = sum(
        row["absolute_difference"] > atol + rtol * abs(row["torch_score"])
        for row in rows
    )

    report = {
        "status": "diagnostic_only",
        "precision_mode": args.precision,
        "checkpoint_sha256": checkpoint_sha,
        "onnx_sha256": onnx_sha,
        "checked_images": len(rows),
        "batch_size": 8,
        "blur_backend": "original_2d",
        "threshold": predictor.threshold,
        "torch_matmul_precision": torch.backends.cuda.matmul.fp32_precision,
        "torch_conv_precision": torch.backends.cudnn.conv.fp32_precision,
        "ort_use_tf32": False,
        "atol": atol,
        "rtol": rtol,
        "max_score_difference": max(row["absolute_difference"] for row in rows),
        "scores_outside_original_tolerance": score_outside,
        "changed_decisions": changed_decisions,
        "torch_vs_saved_max_score_difference": max(
            abs(row["torch_score"] - saved_scores[row["path"]])
            for row in rows
        ),
        "onnx_vs_saved_max_score_difference": max(
            abs(row["onnx_score"] - saved_scores[row["path"]])
            for row in rows
        ),
        "torch_vs_saved_changed_decisions": sum(
            row["torch_alarm"]
            != (saved_scores[row["path"]] > predictor.threshold)
            for row in rows
        ),
        "onnx_vs_saved_changed_decisions": sum(
            row["onnx_alarm"]
            != (saved_scores[row["path"]] > predictor.threshold)
            for row in rows
        ),
        "minimum_reference_threshold_margin": min(
            abs(row["torch_score"] - predictor.threshold) for row in rows
        ),
        "map_pixels": int(map_pixels),
        "max_map_difference": map_max,
        "mean_map_difference": map_abs_sum / map_pixels,
        "relative_map_l2_error": (
            map_squared_error / max(map_squared_reference, 1e-24)
        ) ** 0.5,
        "map_pixels_outside_original_tolerance": map_outside,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = export_dir / f"pipeline_diagnostic_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, indent=2))
    print("Output directory:", output_dir)
    print("PIPELINE DIAGNOSTIC COMPLETED")


if __name__ == "__main__":
    main()
