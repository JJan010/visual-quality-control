"""Sprawdzenie błędnych danych wejściowych i dalszej pracy API."""

import json
import math
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from PIL import Image


BASE_URL = "http://127.0.0.1:8000"


def upload(data: bytes, filename: str) -> tuple[int, dict]:
    boundary = uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    body += data
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    request = Request(
        f"{BASE_URL}/predict",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        with error:
            return error.code, json.load(error)


def main() -> None:
    # Poprawny obraz, ale w formacie nieobsługiwanym przez nasze API.
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="BMP")

    cases = [
        ("empty_file", b"", "empty.png", 400),
        ("invalid_image", b"This is not an image.", "fake.png", 400),
        ("unsupported_format", buffer.getvalue(), "image.bmp", 415),
        (
            "file_too_large",
            b"x" * (10 * 1024 * 1024 + 1),
            "large.png",
            413,
        ),
    ]

    results = []

    for name, data, filename, expected_status in cases:
        status, response = upload(data, filename)
        assert status == expected_status, (
            name, expected_status, status, response
        )
        assert "detail" in response, (name, response)

        results.append({
            "name": name,
            "expected_status": expected_status,
            "actual_status": status,
            "response": response,
        })
        print(f"{name}: HTTP {status} — OK")

    # Po błędnych żądaniach model nadal musi obsłużyć poprawny obraz.
    image_path = Path("data/mvtec_ad/bottle/test/good/017.png")
    status, prediction = upload(image_path.read_bytes(), image_path.name)

    assert status == 200, (status, prediction)
    assert prediction["is_anomaly"] is False, prediction
    assert math.isfinite(prediction["score"]), prediction
    assert prediction["score"] <= prediction["threshold"], prediction

    print("valid_image_after_errors: HTTP 200 — NORMAL — OK")

    with urlopen(f"{BASE_URL}/health", timeout=10) as response:
        health = json.load(response)

    assert health["status"] == "ready", health
    assert health["busy"] is False, health

    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_checks": results,
        "prediction_after_errors": prediction,
        "health_after_checks": health,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("artifacts/api_checks") / f"errors_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Report:", report_path)
    print("API ERROR HANDLING: OK")


if __name__ == "__main__":
    main()
