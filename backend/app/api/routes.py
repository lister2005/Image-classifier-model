import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from app.core.config import MAX_UPLOAD_BYTES, SUPPORTED_CONTENT_TYPES
from app.schemas.prediction import ModelStatus, PredictionResponse
from app.services.image_service import InvalidImageError, decode_image
from app.services.inference_service import Prediction

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    models = request.app.state.model_status
    ready = all(model.ready for model in models.values())
    return {
        "status": "ok" if ready else "degraded",
        "models": {name: model.ready for name, model in models.items()},
    }


@router.get("/models", response_model=dict[str, ModelStatus])
def get_models(request: Request) -> dict[str, ModelStatus]:
    return request.app.state.model_status


@router.post("/predict/skin", response_model=PredictionResponse)
@router.post("/predict/cancer", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> Prediction:
    domain = request.url.path.rsplit("/", maxsplit=1)[-1]
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Upload a JPEG or PNG image.",
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")

    try:
        image = decode_image(contents, file.content_type)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()

    predictor = request.app.state.predictors.get(domain)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"The {domain} model is not configured or could not be loaded.",
        )

    try:
        return await asyncio.to_thread(predictor.predict, image)
    except Exception as exc:
        logger.exception("Inference failed for model domain %s", domain)
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc
