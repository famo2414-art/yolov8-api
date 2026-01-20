import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency", ["method", "path"]
)
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds", "YOLO inference latency"
)
EXCEPTIONS = Counter("exceptions_total", "Unhandled exceptions", ["type"])
APP_READY = Gauge("app_ready", "Readiness flag (1=ready)")

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception as e:
            EXCEPTIONS.labels(type=e.__class__.__name__).inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            REQUEST_LATENCY.labels(request.method, request.url.path).observe(elapsed)
        REQUEST_COUNT.labels(
            request.method, request.url.path, str(getattr(response, "status_code", 500))
        ).inc()
        return response
