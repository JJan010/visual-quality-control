"""Sprawdź maskę i kontur API bez zmiany decyzji klasyfikacji."""

import argparse
import base64
import hashlib
import json
import math
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from check_api_visualization import upload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--calibration-file", type=Path, default=Path(
        "artifacts/runs/patchcore_20261003_135714_557111/"
        "localization_calibration/20261003_182936_190431/threshold.json"
    ))
    args = parser.parse_args()
    raw = args.calibration_file.read_bytes()
    calibration = json.loads(raw)
    calibration_sha = hashlib.sha256(raw).hexdigest()
    results = []

    for relative in ("broken_large/011.png", "good/017.png"):
        path = Path("data/mvtec_ad/bottle/test") / relative
        plain = upload(args.base_url, path, False)
        rendered = upload(args.base_url, path, True)
        assert plain["is_anomaly"] == rendered["is_anomaly"]
        assert plain["threshold"] == rendered["threshold"]
        assert math.isclose(plain["score"], rendered["score"], rel_tol=1e-4, abs_tol=1e-4)
        visual = rendered["visualization"]
        region = visual["localization"]
        assert region is not None, "Restart with VQC_LOCALIZATION_CALIBRATION set."
        assert region["calibration_sha256"] == calibration_sha
        assert region["threshold"] == calibration["threshold"]
        assert region["decision_rule"] == "anomaly_map > threshold"

        with Image.open(BytesIO(base64.b64decode(region["mask_png_base64"], validate=True))) as image:
            assert image.format == "PNG" and image.mode == "L"
            assert image.size == (visual["width"], visual["height"])
            mask = np.array(image)
        assert set(np.unique(mask)).issubset({0, 255})
        count = int(np.count_nonzero(mask))
        assert count == region["mask_pixels"]
        assert mask.size == region["total_pixels"]
        assert math.isclose(count / mask.size, region["mask_fraction"], abs_tol=1e-12)

        with Image.open(BytesIO(base64.b64decode(region["contour_png_base64"], validate=True))) as image:
            assert image.format == "PNG" and image.mode == "RGBA"
            assert image.size == (visual["width"], visual["height"])
            alpha = np.array(image)[..., 3]
        assert bool(np.any(alpha)) == bool(count)
        assert bool(count) == (visual["map_max"] > region["threshold"])
        print(f"{relative}: image anomaly={rendered['is_anomaly']}; region pixels={count}; parity OK")
        results.append({
            "image": relative,
            "image_is_anomaly": rendered["is_anomaly"],
            "region_pixels": count,
            "region_fraction": region["mask_fraction"],
            "score_difference": abs(plain["score"] - rendered["score"]),
        })

    directory = Path("artifacts/api_checks") / (
        "localization_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "passed", "calibration_sha256": calibration_sha,
              "localization_threshold": calibration["threshold"], "results": results}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Report:", directory / "report.json")
    print("API LOCALIZATION: OK")


if __name__ == "__main__":
    main()
