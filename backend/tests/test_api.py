from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.services.inference_service import Prediction


class FakePredictor:
    classes = ["class-a", "class-b"]

    def predict(self, image: Image.Image) -> Prediction:
        assert image.mode == "RGB"
        return Prediction(
            predicted_class="class-a",
            confidence=0.75,
            probabilities={"class-a": 0.75, "class-b": 0.25},
            gradcam=None,
        )


def image_bytes(image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color="red").save(buffer, format=image_format)
    return buffer.getvalue()


def test_health_and_models_report_loaded_predictor() -> None:
    with TestClient(create_app({"skin": FakePredictor()})) as client:
        health = client.get("/health")
        models = client.get("/models")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "models": {"skin": True, "cancer": False}}
    assert models.json()["skin"] == {
        "ready": True,
        "classes": ["class-a", "class-b"],
        "detail": None,
    }
    assert models.json()["cancer"]["ready"] is False


def test_skin_prediction_matches_frontend_response_contract() -> None:
    with TestClient(create_app({"skin": FakePredictor()})) as client:
        response = client.post(
            "/predict/skin",
            files={"file": ("scan.png", image_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "class": "class-a",
        "confidence": 0.75,
        "probabilities": {"class-a": 0.75, "class-b": 0.25},
        "gradcam": None,
    }


def test_cancer_prediction_uses_cancer_model() -> None:
    with TestClient(create_app({"cancer": FakePredictor()})) as client:
        response = client.post(
            "/predict/cancer",
            files={"file": ("scan.jpg", image_bytes("JPEG"), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["class"] == "class-a"


def test_unconfigured_model_returns_service_unavailable() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/predict/skin",
            files={"file": ("scan.png", image_bytes(), "image/png")},
        )

    assert response.status_code == 503


def test_rejects_unsupported_media_type() -> None:
    with TestClient(create_app({"skin": FakePredictor()})) as client:
        response = client.post(
            "/predict/skin",
            files={"file": ("scan.gif", b"gif-data", "image/gif")},
        )

    assert response.status_code == 415


def test_rejects_corrupt_image_and_mismatched_content_type() -> None:
    with TestClient(create_app({"skin": FakePredictor()})) as client:
        corrupt = client.post(
            "/predict/skin",
            files={"file": ("scan.png", b"not an image", "image/png")},
        )
        mismatch = client.post(
            "/predict/skin",
            files={"file": ("scan.png", image_bytes("JPEG"), "image/png")},
        )

    assert corrupt.status_code == 400
    assert mismatch.status_code == 400


def test_rejects_upload_larger_than_ten_megabytes() -> None:
    large_payload = b"x" * (10 * 1024 * 1024 + 1)
    with TestClient(create_app({"skin": FakePredictor()})) as client:
        response = client.post(
            "/predict/skin",
            files={"file": ("large.png", large_payload, "image/png")},
        )

    assert response.status_code == 413
