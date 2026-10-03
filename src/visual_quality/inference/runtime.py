"""Tworzenie predyktora na podstawie konfiguracji aplikacji."""

import json
from pathlib import Path

from visual_quality.inference.backends import configure_ieee_fp32
from visual_quality.inference.patchcore import PatchcorePredictor


def load_predictor(config_path: str | Path) -> PatchcorePredictor:
    """Wczytaj konfigurację i utwórz jeden gotowy do pracy predyktor."""
    config_path = Path(config_path).expanduser().resolve()

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Konfiguracja musi być obiektem JSON.")

    required = {
        "schema_version",
        "backend",
        "device",
        "precision",
        "blur_backend",
        "run_dir",
    }
    allowed = required | {"export_dir", "engine_dir"}

    missing = required - config.keys()
    unknown = config.keys() - allowed

    if missing:
        raise ValueError(f"Brak wymaganych pól: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Nieznane pola konfiguracji: {sorted(unknown)}")

    if config["schema_version"] != 1:
        raise ValueError("Nieobsługiwana wersja konfiguracji.")

    for key in config.keys() - {"schema_version"}:
        value = config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Pole {key} musi być niepustym tekstem.")

    backend = config["backend"]
    if backend not in ("pytorch", "onnx", "tensorrt"):
        raise ValueError(f"Nieznany backend: {backend}")

    if config["blur_backend"] not in ("original_2d", "separable_1d"):
        raise ValueError("Nieznany wariant wygładzania.")

    # Pierwsza wersja aplikacji korzysta z jednego sprawdzonego
    # urządzenia i jawnej polityki precyzji.
    if config["device"] != "cuda:0":
        raise ValueError("Ta konfiguracja aplikacji wymaga cuda:0.")
    if config["precision"] != "ieee":
        raise ValueError("Ta konfiguracja aplikacji wymaga precision=ieee.")

    artifact_fields = set(config) & {"export_dir", "engine_dir"}
    expected_fields = {
        "pytorch": set(),
        "onnx": {"export_dir"},
        "tensorrt": {"engine_dir"},
    }[backend]

    if artifact_fields != expected_fields:
        raise ValueError(
            f"Backend {backend} wymaga pól artefaktów: "
            f"{sorted(expected_fields)}; otrzymano: {sorted(artifact_fields)}"
        )

    def resolve_directory(key: str) -> Path:
        path = Path(config[key]).expanduser()

        if not path.is_absolute():
            path = config_path.parent / path

        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Nie znaleziono katalogu {key}: {path}")
        return path

    run_dir = resolve_directory("run_dir")
    artifact_options = {
        key: resolve_directory(key)
        for key in sorted(expected_fields)
    }

    # Ustawiamy precyzję raz, podczas uruchamiania aplikacji.
    configure_ieee_fp32()

    # Sprawdzenie checkpointu, progu i artefaktów pozostaje
    # odpowiedzialnością istniejącego predyktora.
    return PatchcorePredictor(
        run_dir,
        device=config["device"],
        backend=backend,
        precision=config["precision"],
        blur_backend=config["blur_backend"],
        **artifact_options,
    )
