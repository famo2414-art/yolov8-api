import base64
from pathlib import Path
import pytest
import httpx
from app.main import app

ASSET = Path(__file__).parent / "assets" / "dog.jpg"


@pytest.mark.asyncio
async def test_predict_json_b64():
    data = base64.b64encode(ASSET.read_bytes()).decode()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/predict:json", json={"image_b64": data, "conf": 0.25})
    assert r.status_code == 200
    js = r.json()
    assert "detections" in js and isinstance(js["detections"], list)


@pytest.mark.asyncio
async def test_predict_file():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        with ASSET.open("rb") as f:
            files = {"file": ("dog.jpg", f, "image/jpeg")}
            r = await ac.post("/v1/predict", files=files)
    assert r.status_code == 200
