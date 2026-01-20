from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: Box


class PredictResponse(BaseModel):
    model: str
    time_ms: float
    detections: List[Detection]


class PredictJsonRequest(BaseModel):
    # Provide exactly one of these
    image_b64: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    conf: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def config_overrides(self) -> dict:
        out = {}
        if self.conf is not None:
            out["conf"] = self.conf
        if self.iou is not None:
            out["iou"] = self.iou
        return out
