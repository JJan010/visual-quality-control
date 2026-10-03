import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from check_patchcore_onnx_iobinding import check_close, ATOL, RTOL
from evaluate_patchcore import BottleTestDataset
from visual_quality.inference.backends import configure_ieee_fp32
from visual_quality.inference.patchcore import PatchcorePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()

    torch.cuda.set_device(0)
    configure_ieee_fp32()

    backend_options = {
        "pytorch": {},
        "onnx": {"export_dir": args.export_dir},
        "tensorrt": {"engine_dir": args.engine_dir},
    }
    predictors = {
        name: PatchcorePredictor(
            run_dir,
            device="cuda:0",
            backend=name,
            precision="ieee",
            blur_backend="original_2d",
            **options,
        )
        for name, options in backend_options.items()
    }

    for name, predictor in predictors.items():
        print(
            f"{name}: "
            f"{predictor.runtime_metadata()['feature_extractor_class']}"
        )

    split = json.loads(
        (run_dir / "split.json").read_text(encoding="utf-8")
    )
    dataset = BottleTestDataset(
        root / split["dataset_root"],
        predictors["pytorch"].image_size,
    )
    if len(dataset) != 83:
        raise RuntimeError("Expected 83 test images.")

    results = []

    for batch_size in (1, 8):
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, num_workers=0,
        )
        stats = {
            name: {
                "max_score_difference": 0.0,
                "max_map_difference": 0.0,
                "changed_decisions": 0,
            }
            for name in ("onnx", "tensorrt")
        }
        checked_images = 0

        for index, batch in enumerate(loader, start=1):
            reference = predictors["pytorch"].predict_batch(batch["image"])

            for name in stats:
                candidate = predictors[name].predict_batch(batch["image"])
                item = stats[name]

                item["max_score_difference"] = max(
                    item["max_score_difference"],
                    check_close(candidate.scores, reference.scores),
                )
                item["max_map_difference"] = max(
                    item["max_map_difference"],
                    check_close(
                        candidate.anomaly_maps, reference.anomaly_maps
                    ),
                )
                if candidate.labels.dtype != torch.bool:
                    raise RuntimeError("Expected boolean labels.")

                item["changed_decisions"] += int(
                    (candidate.labels != reference.labels).sum().item()
                )

            checked_images += len(batch["path"])
            if index % 10 == 0 or index == len(loader):
                print(
                    f"B={batch_size}: batch {index}/{len(loader)}",
                    flush=True,
                )

        if checked_images != 83:
            raise RuntimeError("Incorrect number of checked images.")
        if any(item["changed_decisions"] for item in stats.values()):
            raise RuntimeError("Decisions changed after integration.")

        result = {
            "batch_size": batch_size,
            "checked_images": checked_images,
            "comparisons_vs_pytorch": stats,
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    report = {
        "status": "passed",
        "scope": "Unified predictor, IEEE FP32, original_2d blur",
        "atol": ATOL,
        "rtol": RTOL,
        "runtime": {
            name: predictor.runtime_metadata()
            for name, predictor in predictors.items()
        },
        "results": results,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = run_dir / "verification" / f"unified_backends_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Output directory:", output_dir)
    print("UNIFIED PATCHCORE BACKENDS: OK")


if __name__ == "__main__":
    main()
