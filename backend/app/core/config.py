import os
from pathlib import Path

API_TITLE = "MEDSCAN API"
API_VERSION = "1.0.0"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_CONTENT_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.getenv("MEDSCAN_MODEL_DIR", BACKEND_ROOT / "models"))


def cors_origins() -> list[str]:
    configured = os.getenv("MEDSCAN_CORS_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]
