"""ONNX Runtime inference service for the fine-tuned SegFormer model."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.services.chunker import ImageChunk

logger = logging.getLogger(__name__)

FEATURE_COLOURS = {
    "building": "#FF4444",
    "road": "#4488FF",
    "utility": "#FFAA00",
    "vegetation": "#44BB44",
    "water": "#00BBFF",
    "unknown": "#888888",
}


class DetectionModel:
    """Singleton wrapper around the exported SegFormer ONNX model."""

    _instance: Optional["DetectionModel"] = None

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.providers: list[str] = []
        self._loaded = False

    @classmethod
    def get(cls) -> "DetectionModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Load the ONNX graph once per backend process."""
        if self._loaded:
            return

        model_path = _resolve_model_path(settings.ONNX_MODEL_PATH)
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if (
            settings.MODEL_DEVICE.lower() == "cuda"
            and "CUDAExecutionProvider" in available
        ):
            providers.insert(0, "CUDAExecutionProvider")

        logger.info("Loading SegFormer ONNX model from %s", model_path)
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = providers
        self._loaded = True
        logger.info("ONNX model loaded with providers: %s", providers)

    def infer_chunk(self, chunk: ImageChunk) -> List[dict]:
        """
        Run SegFormer on one chunk and return contour polygons.

        The ONNX model expects RGB pixel values shaped [1, 3, 512, 512] and
        returns class logits shaped [1, 4, 512, 512].
        """
        if not self._loaded:
            self.load()

        pixels = _ensure_rgb(chunk.pixels)
        height, width = pixels.shape[1], pixels.shape[2]
        input_size = int(settings.MODEL_INPUT_SIZE)

        input_pixels = _resize_chw(pixels, (input_size, input_size))
        input_pixels = input_pixels[np.newaxis, ...].astype(np.float32, copy=False)

        logits = self.session.run(
            [self.output_name],
            {self.input_name: np.ascontiguousarray(input_pixels)},
        )[0][0]

        if logits.shape[1:] != (height, width):
            logits = _resize_chw(logits, (height, width))

        pred_np = np.argmax(logits, axis=0).astype(np.uint8)
        return _detections_from_prediction(pred_np, chunk)


def _detections_from_prediction(pred_np: np.ndarray, chunk: ImageChunk) -> list[dict]:
    """Convert a class mask into GeoJSON-compatible contour detections."""
    import cv2

    height, width = pred_np.shape
    detections = []

    for feature_name, class_ids in settings.FEATURE_CLASS_MAP.items():
        if not class_ids:
            continue

        mask = np.isin(pred_np, class_ids).astype(np.uint8)
        mask = _core_only_mask(mask, chunk)
        if mask.sum() == 0:
            continue

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < settings.MIN_SEGMENT_AREA_PX:
                continue

            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = max(1.0, 0.002 * perimeter)
            contour = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(contour) < 3:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            geo_poly = _contour_to_geo_polygon(contour, chunk.geo_transform)
            if len(geo_poly) < 4:
                continue

            confidence = float(area) / (width * height)
            detections.append({
                "feature_type": feature_name,
                "confidence": round(min(confidence * 20, 0.99), 4),
                "chunk_id": chunk.chunk_id,
                "pixel_bbox": [int(x), int(y), int(x + w), int(y + h)],
                "geo_polygon": geo_poly,
                "crs": chunk.crs,
                "area_px": int(area),
                "colour": FEATURE_COLOURS.get(feature_name, "#888888"),
            })

    return detections


def _ensure_rgb(pixels: np.ndarray) -> np.ndarray:
    """Return exactly three CHW bands as float32."""
    pixels = pixels.astype(np.float32, copy=False)
    if pixels.shape[0] >= 3:
        return pixels[:3]

    padding = np.zeros(
        (3 - pixels.shape[0], pixels.shape[1], pixels.shape[2]),
        dtype=np.float32,
    )
    return np.concatenate([pixels, padding], axis=0)


def _resize_chw(array: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize a CHW tensor with bilinear interpolation."""
    import cv2

    height, width = size_hw
    resized = [
        cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
        for channel in array
    ]
    return np.stack(resized, axis=0).astype(np.float32, copy=False)


def _contour_to_geo_polygon(contour: np.ndarray, geo_transform: list) -> list:
    """Convert an OpenCV contour to a closed polygon in the source CRS."""
    polygon = []
    for point in contour.reshape(-1, 2):
        col, row = int(point[0]), int(point[1])
        polygon.append(_px_to_geo(col, row, geo_transform))

    if polygon and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon


def _core_only_mask(mask: np.ndarray, chunk: ImageChunk) -> np.ndarray:
    """Keep only the non-overlap tile core to avoid duplicate detections."""
    chunk_size = settings.CHUNK_SIZE
    core_col_off = chunk.col * chunk_size
    core_row_off = chunk.row * chunk_size
    core_width = min(chunk_size, chunk.image_width - core_col_off)
    core_height = min(chunk_size, chunk.image_height - core_row_off)
    if core_width <= 0 or core_height <= 0:
        return np.zeros_like(mask)

    x0 = int(round(core_col_off - chunk.window.col_off))
    y0 = int(round(core_row_off - chunk.window.row_off))
    x1 = min(mask.shape[1], x0 + core_width)
    y1 = min(mask.shape[0], y0 + core_height)
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x0 >= x1 or y0 >= y1:
        return np.zeros_like(mask)

    core = np.zeros_like(mask)
    core[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return core


def _px_to_geo(col: int, row: int, geo_transform: list) -> list:
    """Convert one pixel coordinate to source CRS coordinates."""
    x = geo_transform[0] + col * geo_transform[1] + row * geo_transform[2]
    y = geo_transform[3] + col * geo_transform[4] + row * geo_transform[5]
    return [round(x, 8), round(y, 8)]


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    candidates = [path, Path.cwd() / path]

    service_path = Path(__file__).resolve()
    for parent in service_path.parents:
        candidates.append(parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates[:5])
    raise FileNotFoundError(f"ONNX model not found: {model_path}. Searched: {searched}")


detection_model = DetectionModel.get()
