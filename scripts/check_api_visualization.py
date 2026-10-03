"""Porównaj predykcję API bez wizualizacji i z wizualizacją."""

import argparse
import base64
import json
import math
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

from PIL import Image


def upload(base_url: str, path: Path, include_visualization: bool) -> dict:
    boundary = uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    option = "true" if include_visualization else "false"
    request = Request(
        f"{base_url}/predict?include_visualization={option}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    results = []
    cases = [("broken_large/011.png", True), ("good/017.png", False)]

    for relative, expected in cases:
        path = Path("data/mvtec_ad/bottle/test") / relative
        plain = upload(args.base_url, path, False)
        displayed = upload(args.base_url, path, True)
        assert plain["visualization"] is None
        for response in (plain, displayed):
            assert response["is_anomaly"] is expected, response
            assert math.isfinite(response["score"])
            assert response["is_anomaly"] == (response["score"] > response["threshold"])
            assert all(math.isfinite(v) and v >= 0 for v in response["timings"].values())
        for key in ("threshold", "checkpoint_sha256", "backend", "blur_backend"):
            assert plain[key] == displayed[key], key
        assert math.isclose(plain["score"], displayed["score"], rel_tol=1e-4, abs_tol=1e-4)

        visual = displayed["visualization"]
        assert visual["normalization"] == "per_image_min_max"
        assert visual["coordinates"] == "resized_model_input"
        assert visual["width"] == visual["height"] == 256
        assert math.isfinite(visual["map_min"]) and math.isfinite(visual["map_max"])
        assert visual["map_max"] >= visual["map_min"]
        for key in ("input_png_base64", "heatmap_png_base64"):
            raw = base64.b64decode(visual[key], validate=True)
            with Image.open(BytesIO(raw)) as image:
                assert image.format == "PNG" and image.mode == "RGB"
                assert image.size == (visual["width"], visual["height"])
                image.load()
        difference = abs(plain["score"] - displayed["score"])
        results.append({"image": relative, "score_difference": difference,
                        "is_anomaly": expected, "timings": displayed["timings"],
                        "request_id": displayed["request_id"]})
        print(f"{relative}: score difference={difference:.10f}; PNGs, timing and decision OK")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = Path("artifacts/api_checks") / f"visualization_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "passed", "results": results}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Report:", directory / "report.json")
    print("API VISUALIZATION: OK")


if __name__ == "__main__":
    main()
