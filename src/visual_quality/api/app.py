"""Lokalne API do analizy pojedynczych obrazów."""

import asyncio
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from visual_quality.inference.runtime import load_predictor


LOGGER = logging.getLogger("uvicorn.error")

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


class ImageInputError(Exception):
    """Błąd obrazu przesłanego przez klienta."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class PredictionResponse(BaseModel):
    request_id: str
    score: float
    threshold: float
    is_anomaly: bool
    decision_rule: str
    original_width: int
    original_height: int
    backend: str
    blur_backend: str
    checkpoint_sha256: str


class ModelWorker:
    """Jeden wątek jest właścicielem modelu i jego wywołań."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="visual-quality-gpu",
        )
        self.predictor = None
        self.metadata = {}
        self.busy = False

    def load(self, config_path: str) -> None:
        # Ta metoda działa w tym samym wątku co inferencja.
        torch.cuda.set_device(0)
        self.predictor = load_predictor(config_path)
        self.metadata = self.predictor.runtime_metadata()
        LOGGER.info(
            "Model loaded: backend=%s, blur=%s",
            self.metadata["backend"],
            self.metadata["blur_backend"],
        )

    def unload(self) -> None:
        if self.predictor is not None:
            torch.cuda.synchronize(0)
            self.predictor = None

    def predict_sync(self, data: bytes) -> PredictionResponse:
        # Błędy odczytu obrazu oddzielamy od błędów modelu.
        try:
            with Image.open(BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise ImageInputError(
                        "Supported image formats: PNG, JPEG.",
                        status_code=415,
                    )

                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ImageInputError(
                        "Image exceeds the 16 megapixel limit.",
                        status_code=413,
                    )

                if getattr(image, "n_frames", 1) != 1:
                    raise ImageInputError("Multi-frame images are unsupported.")

                image.load()
                tensor = self.predictor.prepare_image(image)

        except ImageInputError:
            raise
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as error:
            raise ImageInputError(
                "The uploaded file is not a readable image."
            ) from error

        prediction = self.predictor.predict_batch(tensor.unsqueeze(0))

        score = float(prediction.scores.detach().cpu().item())
        is_anomaly = bool(prediction.labels.detach().cpu().item())

        if not math.isfinite(score):
            raise RuntimeError("Model returned a non-finite score.")

        return PredictionResponse(
            request_id=str(uuid4()),
            score=score,
            threshold=self.predictor.threshold,
            is_anomaly=is_anomaly,
            decision_rule="score > threshold",
            original_width=width,
            original_height=height,
            backend=self.metadata["backend"],
            blur_backend=self.metadata["blur_backend"],
            checkpoint_sha256=self.metadata["checkpoint_sha256"],
        )

    def finished(self, future: asyncio.Future) -> None:
        # Zajętość zwalniamy dopiero po rzeczywistym zakończeniu zadania.
        self.busy = False
        if not future.cancelled():
            future.exception()

    async def predict(self, data: bytes) -> PredictionResponse:
        if self.busy:
            raise HTTPException(
                status_code=429,
                detail="Model is busy. Retry shortly.",
                headers={"Retry-After": "1"},
            )

        # Między sprawdzeniem i ustawieniem busy nie ma await:
        # oba kroki wykonują się razem w pętli zdarzeń serwera.
        self.busy = True
        loop = asyncio.get_running_loop()

        try:
            future = loop.run_in_executor(
                self.executor, self.predict_sync, data
            )
        except BaseException:
            self.busy = False
            raise

        future.add_done_callback(self.finished)

        # Anulowanie oczekiwania HTTP nie zwalnia zajętego modelu.
        return await asyncio.shield(future)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("VQC_CONFIG")
    if not config_path:
        raise RuntimeError("Set VQC_CONFIG to the runtime configuration path.")

    worker = ModelWorker()
    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(
            worker.executor, worker.load, config_path
        )
        app.state.worker = worker
        yield
    finally:
        # Sprzątanie też odbywa się w wątku modelu, po jego zadaniach.
        try:
            await loop.run_in_executor(worker.executor, worker.unload)
        finally:
            worker.executor.shutdown(wait=True)


app = FastAPI(
    title="Visual Quality Control",
    version="0.1.0",
    description="Local single-image anomaly detection API.",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request):
    worker = request.app.state.worker
    return {
        "status": "ready",
        "busy": worker.busy,
        "backend": worker.metadata["backend"],
        "blur_backend": worker.metadata["blur_backend"],
        "device": worker.metadata["device"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile):
    try:
        # Odczytujemy najwyżej limit + 1 bajt, aby wykryć przekroczenie.
        data = await file.read(MAX_FILE_BYTES + 1)
    finally:
        await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MiB limit.",
        )

    try:
        return await request.app.state.worker.predict(data)
    except ImageInputError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error),
        ) from error
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Inference failed. See server logs.",
        )
