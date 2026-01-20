import io
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image
from ..models import PredictResponse, PredictJsonRequest
from ..config import settings
from ..service import _image_from_b64, _image_from_url, infer

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post("/predict", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="file must be an image")
    raw = await file.read()
    if len(raw) > settings.request_max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="payload too large")
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    model, time_ms, detections = infer(
        image, settings.confidence_threshold, settings.iou_threshold)
    return PredictResponse(model=model, time_ms=time_ms, detections=detections)


@router.post("/predict:json", response_model=PredictResponse)
async def predict_json(req: PredictJsonRequest):
    if (req.image_b64 is None) == (req.image_url is None):
        raise HTTPException(
            400, detail="Provide exactly one of image_b64 or image_url")
    try:
        if req.image_b64:
            image = _image_from_b64(req.image_b64)
        else:
            image = _image_from_url(str(req.image_url))
    except Exception:
        raise HTTPException(400, detail="Could not decode/download image")

    overrides = req.config_overrides()
    conf = overrides.get("conf", settings.confidence_threshold)
    iou = overrides.get("iou", settings.iou_threshold)
    model, time_ms, detections = infer(image, conf, iou)
    return PredictResponse(model=model, time_ms=time_ms, detections=detections)
