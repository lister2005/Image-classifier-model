import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from app.core.config import MODEL_ROOT

logger = logging.getLogger(__name__)
PREPROCESSING_MODES = {"zero_one", "minus_one_one", "raw_0_255"}


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    gradcam: str | None = None


class Predictor(Protocol):
    classes: list[str]

    def predict(self, image: Image.Image) -> Prediction: ...


class ModelNotReadyError(RuntimeError):
    pass


class KerasPredictor:
    def __init__(
        self,
        model: object,
        classes: list[str],
        preprocessing: str,
        input_size: tuple[int, int],
    ) -> None:
        self.model = model
        self.classes = classes
        self.preprocessing = preprocessing
        self.input_size = input_size
        self._inference_lock = threading.Lock()

    def predict(self, image: Image.Image) -> Prediction:
        resized = image.resize(self.input_size, Image.Resampling.BILINEAR)
        batch = np.asarray(resized, dtype=np.float32)[np.newaxis, ...]
        if self.preprocessing == "zero_one":
            batch /= 255.0
        elif self.preprocessing == "minus_one_one":
            batch = batch / 127.5 - 1.0

        with self._inference_lock:
            output = self.model(batch, training=False)
        scores = np.asarray(output).reshape(-1)
        if scores.size != len(self.classes):
            raise ValueError("Model output count does not match the configured class labels.")
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Model returned invalid class scores.")
        total = float(scores.sum())
        if total <= 0:
            raise ValueError("Model returned an empty probability distribution.")

        probabilities = scores / total
        index = int(np.argmax(probabilities))
        return Prediction(
            predicted_class=self.classes[index],
            confidence=float(probabilities[index]),
            probabilities={
                label: float(probability)
                for label, probability in zip(self.classes, probabilities, strict=True)
            },
        )


def _load_one(domain: str) -> tuple[Predictor | None, str | None]:
    prefix = f"MEDSCAN_{domain.upper()}"
    model_path = Path(os.getenv(f"{prefix}_MODEL_PATH", MODEL_ROOT / f"{domain}.keras"))
    labels_value = os.getenv(f"{prefix}_CLASSES")
    preprocessing = os.getenv(f"{prefix}_PREPROCESSING")

    if not model_path.is_file():
        return None, f"Model artifact not found: {model_path}"
    if not labels_value or not preprocessing:
        return None, (
            f"Set {prefix}_CLASSES and {prefix}_PREPROCESSING to match the trained model."
        )
    if preprocessing not in PREPROCESSING_MODES:
        return None, (
            f"{prefix}_PREPROCESSING must be one of: {', '.join(sorted(PREPROCESSING_MODES))}."
        )

    try:
        classes = json.loads(labels_value)
    except json.JSONDecodeError as exc:
        return None, f"{prefix}_CLASSES must be a JSON array of class labels: {exc}"
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(label, str) or not label for label in classes)
        or len(set(classes)) != len(classes)
    ):
        return None, f"{prefix}_CLASSES must contain unique, non-empty string labels."

    try:
        from tensorflow.keras.models import load_model

        model = load_model(model_path)
    except ImportError:
        return None, "TensorFlow is required to load configured Keras model artifacts."
    except Exception as exc:
        logger.exception("Could not load %s model from %s", domain, model_path)
        return None, f"Model artifact could not be loaded: {exc}"

    shape = model.input_shape
    if isinstance(shape, list):
        return None, "Only single-input image classification models are supported."
    height, width = shape[1:3]
    input_size = (int(width or 224), int(height or 224))
    return KerasPredictor(model, classes, preprocessing, input_size), None


def load_predictors() -> tuple[dict[str, Predictor], dict[str, dict[str, object]]]:
    predictors: dict[str, Predictor] = {}
    status: dict[str, dict[str, object]] = {}
    for domain in ("skin", "cancer"):
        predictor, detail = _load_one(domain)
        if predictor is not None:
            predictors[domain] = predictor
            status[domain] = {"ready": True, "classes": predictor.classes, "detail": None}
        else:
            logger.warning("%s model is not ready: %s", domain, detail)
            status[domain] = {"ready": False, "classes": [], "detail": detail}
    return predictors, status
