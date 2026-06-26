"""Application configuration via environment variables."""

import os
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

    # Optional masked-patch classifiers for detected objects.
    ROOF_CLASSIFIER_ONNX_PATH: str = "models/best_root_top.onnx"
    ROOF_CLASSIFIER_CLASSES_PATH: str = "models/best_root_top.classes.json"
    ROAD_CLASSIFIER_ONNX_PATH: str = "models/best_road.onnx"
    ROAD_CLASSIFIER_CLASSES_PATH: str = "models/best_road.classes.json"
    CLASSIFIER_DEVICE: str = MODEL_DEVICE

    # Detection classes mapped to rural features
    # Fine-tuned SegFormer labels: 0=background, 1=building, 2=road, 3=water
    FEATURE_CLASS_MAP: dict = {
        "building": [1],
        "road": [2],
        "water": [3],
    }

    # BuildingSeparator (ABIS) tunables
    # Controls how the watershed-based instance separator behaves.
    # Raise BUILDING_MIN_AREA to suppress small roof fragments.
    # Tighten BUILDING_MAX_DISTANCE / BUILDING_MAX_COLOUR to prevent
    # adjacent buildings from merging into one cluster.
    BUILDING_MIN_AREA: int = 120            # min px area per connected component
    BUILDING_KERNEL_SIZE: int = 3           # morphological structuring element size
    BUILDING_MAX_DISTANCE: float = 10.0     # max centroid/bbox gap to connect components (px)
    BUILDING_MAX_COLOUR: float = 45.0       # max mean-RGB L2 distance to connect components
    BUILDING_MAX_ORIENTATION: float = 25.0  # max ellipse-angle difference to connect (deg)
    BUILDING_MERGE_MAX_COLOUR: float = 45.0         # RAG merge: max colour distance
    BUILDING_MERGE_MIN_BOUNDARY: int = 8            # RAG merge: min shared boundary pixels
    BUILDING_SIDE_OVERLAP_RATIO: float = 0.25        # min side overlap to merge roof rectangles
    BUILDING_ROOF_SPLIT_ENABLED: bool = False       # avoid fragmenting one roof by material
    BUILDING_ROOF_SPLIT_CLUSTERS: int = 2
    BUILDING_ROOF_SPLIT_MIN_AREA: int = 2500
    BUILDING_ROOF_SPLIT_MIN_COLOUR_DISTANCE: float = 95.0
    ROAD_BUILDING_SUPPRESS_PROB: float = 0.35        # road probability wins over building
    ROAD_BUILDING_SUPPRESS_DILATE: int = 2           # pixels around road removed from building mask

    # Adaptive Semantic Road Graph Reconstruction (ASRG)
    ROAD_RECONSTRUCTION_ENABLED: bool = True
    ROAD_SEED_PROBABILITY: float = 0.35
    ROAD_PROBABILITY_DILATE: int = 1
    ROAD_MAX_CONNECT_DISTANCE: int = 40
    ROAD_MAX_ANGLE_DEG: float = 20.0
    ROAD_TEMPLATE_MIN_SCORE: float = 0.50
    ROAD_MIN_PROBABILITY: float = 0.25
    ROAD_MAX_WIDTH_DIFFERENCE: float = 0.30
    ROAD_MAX_PATH_COST: float = 6.0

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 15  # seconds

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
