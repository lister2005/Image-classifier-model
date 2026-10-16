from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import API_TITLE, API_VERSION, cors_origins
from app.schemas.prediction import ModelStatus
from app.services.inference_service import Predictor, load_predictors


def create_app(
    predictors: Mapping[str, Predictor] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        if predictors is None:
            loaded_predictors, loaded_status = load_predictors()
        else:
            loaded_predictors = dict(predictors)
            loaded_status = {
                domain: {
                    "ready": domain in loaded_predictors,
                    "classes": (
                        loaded_predictors[domain].classes if domain in loaded_predictors else []
                    ),
                    "detail": None if domain in loaded_predictors else "Model is not configured.",
                }
                for domain in ("skin", "cancer")
            }
        application.state.predictors = loaded_predictors
        application.state.model_status = {
            domain: ModelStatus(**details) for domain, details in loaded_status.items()
        }
        yield

    application = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
