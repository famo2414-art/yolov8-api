from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = "yolov8-api"
    environment: str = Field(default="dev")
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    request_max_mb: int = 10
    torch_index_url: str = "https://download.pytorch.org/whl/cpu"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
