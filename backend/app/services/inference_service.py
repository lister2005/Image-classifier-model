import base64
import json
import logging
import os
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from app.core.config import MODEL_ROOT

logger = logging.getLogger(__name__)
PREPROCESSING_MODES = {"zero_one", "minus_one_one", "raw_0_255"}
DEFAULT_SKIN_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
GRADCAM_MAX_IMAGE_SIZE = (512, 512)


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


def _normalize_class_scores(output: object, classes: list[str]) -> np.ndarray:
    scores = np.asarray(output, dtype=np.float32).reshape(-1)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Model returned invalid class scores.")

    if len(classes) == 2 and scores.size == 1:
        if scores[0] > 1:
            raise ValueError("Binary model output must be a probability between 0 and 1.")
        return np.array([1.0 - scores[0], scores[0]], dtype=np.float32)
    if scores.size != len(classes):
        raise ValueError("Model output count does not match the configured class labels.")

    total = float(scores.sum())
    if total <= 0:
        raise ValueError("Model returned an empty probability distribution.")
    return scores / total


def _encode_gradcam(
    activations: object,
    gradients: object,
    image: Image.Image,
) -> str:
    feature_map = np.asarray(activations, dtype=np.float32)[0]
    gradient_map = np.asarray(gradients, dtype=np.float32)[0]
    channel_weights = gradient_map.mean(axis=(0, 1))
    heatmap = np.maximum(np.sum(feature_map * channel_weights, axis=-1), 0)
    maximum = float(heatmap.max())
    if maximum > 0:
        heatmap /= maximum

    overlay_base = image.copy()
    overlay_base.thumbnail(GRADCAM_MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap * 255, 0, 255)))
    heatmap_image = heatmap_image.resize(overlay_base.size, Image.Resampling.BILINEAR)
    heat = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    heat_color = np.stack(
        (
            np.clip(1.5 - np.abs(4 * heat - 3), 0, 1),
            np.clip(1.5 - np.abs(4 * heat - 2), 0, 1),
            np.clip(1.5 - np.abs(4 * heat - 1), 0, 1),
        ),
        axis=-1,
    )
    base = np.asarray(overlay_base, dtype=np.float32)
    alpha = (heat * 0.45)[..., np.newaxis]
    blended = np.clip(base * (1 - alpha) + heat_color * 255 * alpha, 0, 255)

    output = BytesIO()
    Image.fromarray(blended.astype(np.uint8)).save(output, format="JPEG", quality=85)
    return base64.b64encode(output.getvalue()).decode("ascii")


class KerasPredictor:
    def __init__(
        self,
        model: object,
        classes: list[str],
        preprocessing: str,
        input_size: tuple[int, int],
        gradcam_layer: str | None = None,
    ) -> None:
        import tensorflow as tf

        self.model = model
        self.classes = classes
        self.preprocessing = preprocessing
        self.input_size = input_size
        self._tf = tf
        self._inference_lock = threading.Lock()
        self._gradcam_model, self._head_layers = self._create_gradcam_model(
            model,
            tf,
            gradcam_layer,
        )

    @staticmethod
    def _is_spatial_layer(layer: object) -> bool:
        shape = getattr(getattr(layer, "output", None), "shape", ())
        return len(shape) == 4

    @classmethod
    def _create_gradcam_model(
        cls,
        model: object,
        tf: object,
        gradcam_layer: str | None,
    ) -> tuple[object, list[object] | None]:
        nested_models = [
            (index, layer)
            for index, layer in enumerate(model.layers)
            if isinstance(layer, tf.keras.Model)
        ]
        if len(nested_models) > 1:
            raise ValueError("Grad-CAM supports a single nested feature extractor.")

        if nested_models:
            index, backbone = nested_models[0]
            if gradcam_layer:
                target_layer = backbone.get_layer(gradcam_layer)
            else:
                target_layer = next(
                    (layer for layer in reversed(backbone.layers) if cls._is_spatial_layer(layer)),
                    None,
                )
            if target_layer is None or not cls._is_spatial_layer(target_layer):
                raise ValueError("No spatial feature layer is available for Grad-CAM.")
            if target_layer is not backbone.layers[-1]:
                raise ValueError(
                    "For nested backbones, Grad-CAM must use the final spatial layer."
                )
            feature_model = tf.keras.Model(
                inputs=backbone.inputs,
                outputs=target_layer.output,
            )
            return feature_model, list(model.layers[index + 1 :])

        if gradcam_layer:
            target_layer = model.get_layer(gradcam_layer)
        else:
            target_layer = next(
                (layer for layer in reversed(model.layers) if cls._is_spatial_layer(layer)),
                None,
            )
        if target_layer is None or not cls._is_spatial_layer(target_layer):
            raise ValueError("No spatial feature layer is available for Grad-CAM.")
        gradcam_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
        return gradcam_model, None

    def predict(self, image: Image.Image) -> Prediction:
        resized = image.resize(self.input_size, Image.Resampling.BILINEAR)
        batch = np.asarray(resized, dtype=np.float32)[np.newaxis, ...]
        if self.preprocessing == "zero_one":
            batch /= 255.0
        elif self.preprocessing == "minus_one_one":
            batch = batch / 127.5 - 1.0

        with self._inference_lock:
            with self._tf.GradientTape() as tape:
                if self._head_layers is None:
                    activations, output = self._gradcam_model([batch], training=False)
                else:
                    activations = self._gradcam_model([batch], training=False)
                    tape.watch(activations)
                    output = activations
                    for layer in self._head_layers:
                        output = layer(output, training=False)

                raw_scores = self._tf.reshape(output, (-1,))
                probabilities = _normalize_class_scores(raw_scores.numpy(), self.classes)
                if raw_scores.shape[0] == 1 and len(self.classes) == 2:
                    probability_tensor = self._tf.stack(
                        (1 - raw_scores[0], raw_scores[0]),
                    )
                else:
                    probability_tensor = raw_scores / self._tf.reduce_sum(raw_scores)
                index = int(np.argmax(probabilities))
                class_score = probability_tensor[index]

            gradients = tape.gradient(class_score, activations)
        if gradients is None:
            raise ValueError("Grad-CAM could not compute gradients for the predicted class.")

        index = int(np.argmax(probabilities))
        return Prediction(
            predicted_class=self.classes[index],
            confidence=float(probabilities[index]),
            probabilities={
                label: float(probability)
                for label, probability in zip(self.classes, probabilities, strict=True)
            },
            gradcam=_encode_gradcam(activations, gradients, image),
        )


def _load_one(domain: str) -> tuple[Predictor | None, str | None]:
    prefix = f"MEDSCAN_{domain.upper()}"
    model_path = Path(os.getenv(f"{prefix}_MODEL_PATH", MODEL_ROOT / f"{domain}.keras"))
    labels_value = os.getenv(f"{prefix}_CLASSES")
    preprocessing = os.getenv(f"{prefix}_PREPROCESSING")

    if not model_path.is_file() and not model_path.is_dir():
        return None, f"Model artifact not found: {model_path}"
    if domain == "skin":
        labels_value = labels_value or json.dumps(DEFAULT_SKIN_CLASSES)
        preprocessing = preprocessing or "minus_one_one"
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
        from tensorflow.keras.models import load_model, model_from_json

        if model_path.is_dir():
            config_path = model_path / "config.json"
            weights_path = model_path / "model.weights.h5"
            if not config_path.is_file() or not weights_path.is_file():
                return None, (
                    "Keras model directory must contain config.json and model.weights.h5."
                )
            model = model_from_json(config_path.read_text(encoding="utf-8"))
            model.load_weights(weights_path)
        else:
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
    output_shape = model.output_shape
    output_count = output_shape[-1] if not isinstance(output_shape, list) else None
    if output_count != len(classes) and not (output_count == 1 and len(classes) == 2):
        return None, (
            f"Model output count ({output_count}) does not match configured labels "
            f"({len(classes)})."
        )

    try:
        predictor = KerasPredictor(
            model,
            classes,
            preprocessing,
            input_size,
            gradcam_layer=os.getenv(f"{prefix}_GRADCAM_LAYER"),
        )
    except (ValueError, TypeError) as exc:
        return None, f"Model does not support the configured Grad-CAM setup: {exc}"
    return predictor, None


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
