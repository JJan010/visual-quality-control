import hashlib
import json
import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from visual_quality.inference.filters import SeparableGaussianBlur2d
from visual_quality.inference.backends import (
    create_feature_backend,
    precision_settings,
    require_ieee_fp32,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PatchcorePrediction:
    """Wyniki pozostają na urządzeniu, na którym wykonano inferencję."""

    scores: torch.Tensor
    labels: torch.Tensor
    anomaly_maps: torch.Tensor


class PatchcorePredictor:
    """Wczytuje zapisany eksperyment i wykonuje inferencję PatchCore."""

    def __init__(
        self,
        run_dir: str | Path,
        device: str = "cuda",
        *,
        blur_backend: str = "original_2d",
        backend: str = "pytorch",
        export_dir: str | Path | None = None,
        engine_dir: str | Path | None = None,
        precision: str | None = None,
    ):
        self.run_dir = Path(run_dir).resolve()
        if blur_backend not in ("original_2d", "separable_1d"):
            raise ValueError(f"Nieznany wariant wygładzania: {blur_backend}")
        self.blur_backend = blur_backend
        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU jest niedostępne.")

        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        if backend not in ("pytorch", "onnx", "tensorrt"):
            raise ValueError(f"Nieznany backend: {backend}")
        self.backend = backend

        self.precision = (
            precision if precision is not None
            else ("inherit" if backend == "pytorch" else "ieee")
        )
        if self.precision not in ("inherit", "ieee"):
            raise ValueError(f"Nieznana polityka precyzji: {self.precision}")

        if backend != "pytorch":
            if self.device != torch.device("cuda:0"):
                raise ValueError("Adaptery aplikacji wymagają cuda:0.")
            if self.precision != "ieee":
                raise ValueError("ONNX i TensorRT wymagają precision='ieee'.")

        if backend == "pytorch" and (
            export_dir is not None or engine_dir is not None
        ):
            raise ValueError("Backend PyTorch nie przyjmuje katalogu eksportu.")
        if backend == "onnx" and (
            export_dir is None or engine_dir is not None
        ):
            raise ValueError("Backend ONNX wymaga wyłącznie export_dir.")
        if backend == "tensorrt" and (
            engine_dir is None or export_dir is not None
        ):
            raise ValueError("Backend TensorRT wymaga wyłącznie engine_dir.")

        if self.precision == "ieee":
            require_ieee_fp32()

        checkpoint_path = self.run_dir / "model.pt"
        calibration_path = self.run_dir / "calibration/threshold.json"

        calibration = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
        self.checkpoint_sha256 = file_sha256(checkpoint_path)

        if self.checkpoint_sha256 != calibration["checkpoint_sha256"]:
            raise RuntimeError("Próg dotyczy innego checkpointu.")
        if calibration["decision_rule"] != "score > threshold":
            raise RuntimeError("Nieobsługiwana reguła decyzji.")
        if calibration["score_definition"] != "anomalib.PatchcoreModel.pred_score":
            raise RuntimeError("Nieobsługiwana definicja wyniku anomalii.")

        self.threshold = float(calibration["threshold"])
        if not math.isfinite(self.threshold):
            raise RuntimeError("Próg musi być skończoną liczbą.")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.config = checkpoint["config"]

        if self.config != calibration["config"]:
            raise RuntimeError("Konfiguracja modelu i kalibracji jest różna.")

        for name, expected in checkpoint["versions"].items():
            actual = version(name)
            if actual != expected:
                raise RuntimeError(
                    f"Inna wersja {name}: {actual}; oczekiwano {expected}."
                )

        self.image_size = self.config["image_size"]

        # Obraz PIL -> tensor RGB float32 o wartościach 0–1.
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(
                (self.image_size, self.image_size),
                antialias=True,
            ),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.normalize = v2.Normalize(
            mean=self.config["normalization_mean"],
            std=self.config["normalization_std"],
        )

        self.model = PatchcoreModel(
            backbone=self.config["backbone"],
            layers=self.config["layers"],
            pre_trained=False,
            num_neighbors=self.config["num_neighbors"],
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        extractor, artifact = create_feature_backend(
            self.model.feature_extractor,
            backend=self.backend,
            device=self.device,
            checkpoint_sha256=self.checkpoint_sha256,
            model_config=self.config,
            export_dir=export_dir,
            engine_dir=engine_dir,
        )
        self.model.feature_extractor = extractor
        self.backend_artifact = artifact

        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        if self.model.memory_bank.shape[0] == 0:
            raise RuntimeError("Bank wzorców jest pusty.")

        # Najpierw wczytujemy pełny oryginalny checkpoint.
        # Dopiero potem wybieramy implementację wykonawczą filtra.
        if self.blur_backend == "separable_1d":
            original = self.model.anomaly_map_generator.blur
            if (
                original.border_type != "reflect"
                or original.padding != "same"
                or original.channels != 1
            ):
                raise RuntimeError("Nieobsługiwana konfiguracja filtra.")

            optimized = SeparableGaussianBlur2d(original.kernel)
            optimized = optimized.to(self.device)
            optimized.eval()
            self.model.anomaly_map_generator.blur = optimized

    def runtime_metadata(self) -> dict:
        """Opis konfiguracji wykonawczej do logów i raportów."""
        return {
            "backend": self.backend,
            "device": str(self.device),
            "precision_policy": self.precision,
            "torch_precision": precision_settings(),
            "blur_backend": self.blur_backend,
            "blur_class": type(self.model.anomaly_map_generator.blur).__name__,
            "feature_extractor_class": type(self.model.feature_extractor).__name__,
            "checkpoint_sha256": self.checkpoint_sha256,
            "threshold": self.threshold,
            "decision_rule": "score > threshold",
            "image_size": self.image_size,
            "artifact": dict(self.backend_artifact),
        }

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Zwraca tensor CPU o kształcie (3, H, W), przed normalizacją."""
        return self.transform(image.convert("RGB"))

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor) -> PatchcorePrediction:
        """Przyjmuje batch RGB float32 w zakresie 0–1, przed normalizacją."""
        expected_shape = (3, self.image_size, self.image_size)

        if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                f"Oczekiwano (B, {expected_shape}), otrzymano {images.shape}."
            )
        if images.shape[0] == 0:
            raise ValueError("Batch nie może być pusty.")
        if images.dtype != torch.float32:
            raise TypeError("Oczekiwano tensora float32.")

        if self.precision == "ieee":
            require_ieee_fp32()

        normalized = self.normalize(images.to(self.device))
        output = self.model(normalized)
        scores = output.pred_score.reshape(-1)

        return PatchcorePrediction(
            scores=scores,
            labels=scores > self.threshold,
            anomaly_maps=output.anomaly_map,
        )
