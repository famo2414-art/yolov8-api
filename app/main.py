from fastapi import FastAPI
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from .config import settings
from .logger import configure_logging
from .instrumentation import MetricsMiddleware, APP_READY
from .routers import predict

configure_logging(settings.log_level)
app = FastAPI(title="YOLOv8 Object Detection API", version="1.0.0")
app.add_middleware(MetricsMiddleware)
app.include_router(predict.router)


@app.on_event("startup")
async def _startup():
    APP_READY.set(1)


@app.on_event("shutdown")
async def _shutdown():
    APP_READY.set(0)


@app.get("/healthz")
async def health():
    return {"status": "ok", "ready": APP_READY._value.get() == 1}


@app.get("/version")
async def version():
    return {"app": settings.app_name, "env": settings.environment, "model": settings.model_name}


@app.get("/metrics")
async def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
