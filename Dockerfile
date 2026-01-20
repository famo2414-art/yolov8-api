FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# ONE 'RUN' here. Do not duplicate.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libgl1 git curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --upgrade pip

FROM base AS deps
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
RUN pip install fastapi uvicorn[standard] gunicorn prometheus-client pydantic-settings httpx pillow requests python-multipart pytest pytest-asyncio
RUN pip install ultralytics opencv-python-headless

FROM base AS runtime
# bring in all installed Python bits from deps
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin/ /usr/local/bin/

# 🔥 Pre-download YOLOv8n weights into the FINAL image
RUN python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')  # cached under /root/.cache (and settings at $YOLO_CONFIG_DIR)
PY

# app code last to keep cache helpful
COPY . /app
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["gunicorn", "app.main:app", "-c", "gunicorn_conf.py"]
