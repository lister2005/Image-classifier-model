from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str = Field(
        validation_alias="predicted_class",
        serialization_alias="class",
    )
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    gradcam: str | None = None


class ModelStatus(BaseModel):
    ready: bool
    classes: list[str] = Field(default_factory=list)
    detail: str | None = None
