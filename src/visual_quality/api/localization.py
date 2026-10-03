"""Kontur z progu mapy skalibrowanego na poprawnych obrazach."""

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel, FiniteFloat


@dataclass(frozen=True)
class LocalizationCalibration:
    threshold: float
    sha256: str


class LocalizationResult(BaseModel):
    threshold: FiniteFloat
    calibration_sha256: str
    decision_rule: str = "anomaly_map > threshold"
    mask_pixels: int
    total_pixels: int
    mask_fraction: FiniteFloat
    mask_png_base64: str
    contour_png_base64: str
    description: str = "Thresholded model map; not ground truth."


def load_localization_calibration(
    path: str | Path, predictor, runtime_config_path: str | Path
) -> LocalizationCalibration:
    """Odrzuć kalibrację innego modelu, filtra lub trybu wykonania."""
    raw = Path(path).expanduser().resolve().read_bytes()
    report = json.loads(raw)
    required = {
        "schema_version": 1,
        "purpose": "localization_threshold",
        "method": "quantile_of_normal_validation_map_maxima",
        "quantile_method": "higher",
        "pixel_decision_rule": "anomaly_map > threshold",
        "batch_size": 1,
        "image_size": predictor.image_size,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise RuntimeError(f"Unsupported localization calibration: {key}.")
    if report["runtime"] != predictor.runtime_metadata():
        raise RuntimeError("Localization calibration does not match the active runtime.")
    for path_to_check, key in (
        (predictor.run_dir / "split.json", "split_sha256"),
        (Path(runtime_config_path).expanduser(), "runtime_config_sha256"),
    ):
        actual = hashlib.sha256(path_to_check.read_bytes()).hexdigest()
        if report[key] != actual:
            raise RuntimeError(f"Localization calibration mismatch: {key}.")

    threshold = float(report["threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise RuntimeError("Invalid localization threshold.")
    return LocalizationCalibration(threshold, hashlib.sha256(raw).hexdigest())


def encode_png(array: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_localization(
    anomaly_map: np.ndarray, calibration: LocalizationCalibration
) -> LocalizationResult:
    if anomaly_map.ndim != 2 or anomaly_map.size == 0:
        raise ValueError("Expected a non-empty 2D anomaly map.")
    if not np.isfinite(anomaly_map).all():
        raise ValueError("Non-finite anomaly map.")

    # Jedyny warunek przynależności piksela do regionu.
    mask = anomaly_map > calibration.threshold

    # Kontur to piksele maski, które nie są otoczone maską ze wszystkich stron.
    # Bez filtrowania małych regionów i bez zmiany skalibrowanej maski.
    height, width = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    interior = np.ones_like(mask)
    for y in range(3):
        for x in range(3):
            interior &= padded[y:y + height, x:x + width]
    boundary = mask & ~interior

    # Ciemna obwódka poprawia widoczność turkusowego konturu na jasnym tle.
    # To zabieg graficzny; zapis maski pozostaje niezmieniony.
    outline = np.asarray(
        Image.fromarray(boundary.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))
    ) > 0
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[outline] = (10, 22, 30, 220)
    rgba[boundary] = (0, 245, 198, 255)
    count = int(mask.sum())
    return LocalizationResult(
        threshold=calibration.threshold,
        calibration_sha256=calibration.sha256,
        mask_pixels=count,
        total_pixels=int(mask.size),
        mask_fraction=count / mask.size,
        mask_png_base64=encode_png(mask.astype(np.uint8) * 255),
        contour_png_base64=encode_png(rgba),
    )
