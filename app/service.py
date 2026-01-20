import base64
import io
import time
from typing import Tuple, Optional
import requests
from PIL import Image
from ultralytics import YOLO
from .config import settings
from .instrumentation import INFERENCE_LATENCY

_model: Optional[YOLO] = None

def load_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(settings.model_name)
    return _model

def _image_from_b64(data: str) -> Image.Image:
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def infer(image: Image.Image, conf: float, iou: float) -> Tuple[str, float, list]:
    model = load_model()
    with INFERENCE_LATENCY.time():
        t0 = time.perf_counter()
        results = model.predict(image, conf=conf, iou=iou, verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0
    name = getattr(model, "ckpt_path", settings.model_name).split("/")[-1]
    detections = []
    for r in results:
        for b in r.boxes:
            conf = float(b.conf.item())
            xyxy = b.xyxy[0].tolist()
            cls_idx = int(b.cls.item())
            cls_name = r.names.get(cls_idx, str(cls_idx))
            detections.append({
                "class_name": cls_name,
                "confidence": conf,
                "box": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
            })
    return name, dt, detections
