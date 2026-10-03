"""Wizualizacja wyników: nie zmienia wyniku ani progu modelu."""

import base64
from io import BytesIO

import numpy as np
from matplotlib import colormaps
from PIL import Image
from pydantic import BaseModel, FiniteFloat

from visual_quality.api.localization import (
    LocalizationCalibration, LocalizationResult, create_localization,
)


class Timings(BaseModel):
    preprocessing_ms: FiniteFloat
    inference_ms: FiniteFloat
    visualization_ms: FiniteFloat


class Visualization(BaseModel):
    input_png_base64: str
    heatmap_png_base64: str
    width: int
    height: int
    map_min: FiniteFloat
    map_max: FiniteFloat
    normalization: str = "per_image_min_max"
    colormap: str = "inferno"
    coordinates: str = "resized_model_input"
    localization: LocalizationResult | None = None


def png_base64(pixels: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_visualization(
    image: np.ndarray, anomaly_map: np.ndarray,
    calibration: LocalizationCalibration | None = None,
) -> Visualization:
    """Przyjmuje obraz HWC 0–1 i surową mapę HW, oba na CPU."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image.")
    if anomaly_map.shape != image.shape[:2]:
        raise ValueError("Image and anomaly map dimensions differ.")
    if not np.isfinite(image).all() or not np.isfinite(anomaly_map).all():
        raise ValueError("Non-finite visualization input.")

    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())
    span = maximum - minimum
    # Stała mapa ma jednolity kolor, bez dzielenia przez zero.
    normalized = (
        (anomaly_map.astype(np.float64) - minimum) / span
        if span > 0 else np.zeros_like(anomaly_map, dtype=np.float64)
    )
    normalized = np.clip(normalized, 0, 1)
    heatmap = colormaps["inferno"](normalized, bytes=True)[..., :3]
    rgb = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)

    return Visualization(
        input_png_base64=png_base64(rgb),
        heatmap_png_base64=png_base64(heatmap),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        map_min=minimum,
        map_max=maximum,
        localization=(create_localization(anomaly_map, calibration)
                      if calibration is not None else None),
    )
