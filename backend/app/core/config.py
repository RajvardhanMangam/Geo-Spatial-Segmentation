"""Application configuration via environment variables."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # File storage
    UPLOAD_DIR: str = "/tmp/mopr_uploads"
    OUTPUT_DIR: str = "/tmp/mopr_outputs"
    MAX_UPLOAD_SIZE_GB: int = 6

    # Chunking
    CHUNK_SIZE: int = 1024          # pixels per chunk (width and height)
    CHUNK_OVERLAP: int = 64         # pixel overlap to avoid edge artifacts
    BATCH_SIZE: int = 4             # chunks per GPU batch

    # GeoTIFF preprocessing
    # Modes: percentile, minmax, dtype
    TIFF_PREPROCESS_MODE: str = "percentile"
    TIFF_PERCENTILE_LOW: float = 2.0
    TIFF_PERCENTILE_HIGH: float = 98.0

    # Model
    MODEL_NAME: str = "segformer_epoch_35.onnx"
    ONNX_MODEL_PATH: str = "models/segformer_epoch_35.onnx"
    MODEL_INPUT_SIZE: int = 512
    MODEL_DEVICE: str = "cuda" if os.environ.get("USE_GPU") else "cpu"
    MODEL_CACHE_DIR: str = "/tmp/model_cache"
    MIN_SEGMENT_AREA_PX: int = 100

    # Detection classes mapped to rural features
    # Fine-tuned SegFormer labels: 0=background, 1=building, 2=road, 3=water
    FEATURE_CLASS_MAP: dict = {
        "building": [1],
        "road": [2],
        "water": [3],
    }

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 15  # seconds

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
