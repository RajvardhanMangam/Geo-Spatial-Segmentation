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

    # Model
    MODEL_NAME: str = "facebook/mask2former-swin-large-ade-semantic"
    MODEL_DEVICE: str = "cuda" if os.environ.get("USE_GPU") else "cpu"
    MODEL_CACHE_DIR: str = "/tmp/model_cache"

    # Detection classes mapped to rural features
    # Adjust these indices based on your fine-tuned model's label map
    FEATURE_CLASS_MAP: dict = {
        "building": [1, 2, 3],       # ADE20K: building, house, shelter
        "road": [6, 9, 52],          # ADE20K: road, path, sidewalk
        "utility": [83, 93, 130],    # ADE20K: pole, tower, wire
        "vegetation": [4, 17],       # ADE20K: tree, grass
        "water": [21, 26, 60],       # ADE20K: water, river, pond
    }

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 15  # seconds

    class Config:
        env_file = ".env"


settings = Settings()
