"""ONNX Runtime inference service for the fine-tuned SegFormer model."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.services.building_separator import BuildingSeparator
from app.services.chunker import ImageChunk
from app.services.road_reconstructor import RoadReconstructor

logger = logging.getLogger(__name__)

FEATURE_COLOURS = {
    "building": "#FF4444",
    "road":     "#4488FF",
    "road_added": "#07185c",
    "utility":  "#FFAA00",
    "vegetation": "#44BB44",
    "water":    "#00BBFF",
    "unknown":  "#888888",
}

# Module-level separator instance — shares one kernel allocation across chunks.
_separator = BuildingSeparator(
    min_area=settings.BUILDING_MIN_AREA,
    kernel_size=settings.BUILDING_KERNEL_SIZE,
    max_distance=settings.BUILDING_MAX_DISTANCE,
    max_colour=settings.BUILDING_MAX_COLOUR,
    max_orientation=settings.BUILDING_MAX_ORIENTATION,
    side_overlap_ratio=settings.BUILDING_SIDE_OVERLAP_RATIO,
    merge_max_colour=settings.BUILDING_MERGE_MAX_COLOUR,
    merge_min_shared_boundary=settings.BUILDING_MERGE_MIN_BOUNDARY,
    roof_split_enabled=settings.BUILDING_ROOF_SPLIT_ENABLED,
    roof_split_clusters=settings.BUILDING_ROOF_SPLIT_CLUSTERS,
    roof_split_min_area=settings.BUILDING_ROOF_SPLIT_MIN_AREA,
    roof_split_min_color_distance=settings.BUILDING_ROOF_SPLIT_MIN_COLOUR_DISTANCE,
)

_road_reconstructor = RoadReconstructor(
    max_distance=settings.ROAD_MAX_CONNECT_DISTANCE,
    max_angle=settings.ROAD_MAX_ANGLE_DEG,
    min_probability=settings.ROAD_MIN_PROBABILITY,
    min_template_score=settings.ROAD_TEMPLATE_MIN_SCORE,
    max_width_difference=settings.ROAD_MAX_WIDTH_DIFFERENCE,
    max_path_cost=settings.ROAD_MAX_PATH_COST,
)


class MaskedPatchClassifier:
    """Lazy ONNX classifier for masked RGB object crops."""

    def __init__(
        self,
        model_path: str,
        classes_path: str,
        fallback_prefix: str,
        default_label: str,
    ):
        self.model_path = model_path
        self.classes_path = classes_path
        self.fallback_prefix = fallback_prefix
        self.default_label = default_label
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.classes: list[str] = []
        self.image_size = 224
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._loaded = False
        self._available = True

    def load(self) -> bool:
        if self._loaded:
            return self._available

        self._load_classes_metadata()
        try:
            model_path = _resolve_model_path(self.model_path)

            providers = ["CPUExecutionProvider"]
            available = ort.get_available_providers()
            if (
                settings.CLASSIFIER_DEVICE.lower() == "cuda"
                and "CUDAExecutionProvider" in available
            ):
                providers.insert(0, "CUDAExecutionProvider")

            logger.info("Loading masked-patch classifier from %s", model_path)
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self._available = bool(self.classes)
        except Exception:
            logger.exception("Masked-patch classifier unavailable: %s", self.model_path)
            self._available = False

        self._loaded = True
        return self._available

    def _load_classes_metadata(self) -> None:
        try:
            classes_path = _resolve_model_path(self.classes_path)
            with open(classes_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if isinstance(metadata, list):
                self.classes = [str(label) for label in metadata]
                return
            self.classes = [str(label) for label in metadata.get("classes", [])]
            self.image_size = int(metadata.get("image_size", self.image_size))
            self.mean = np.array(metadata.get("mean", self.mean.tolist()), dtype=np.float32)
            self.std = np.array(metadata.get("std", self.std.tolist()), dtype=np.float32)
        except Exception:
            logger.exception("Classifier class metadata unavailable: %s", self.classes_path)

    def predict(
        self,
        rgb_hwc: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> dict | None:
        if not self.load() or self.session is None:
            return None

        patch = _masked_rgb_patch(rgb_hwc, mask, bbox)
        if patch is None:
            return None

        tensor = _classifier_tensor(patch, self.image_size, self.mean, self.std)
        logits = self.session.run(
            [self.output_name],
            {self.input_name: np.ascontiguousarray(tensor)},
        )[0][0]
        probabilities = _softmax(logits.astype(np.float32), axis=0)
        idx = int(np.argmax(probabilities))
        label = self.classes[idx] if idx < len(self.classes) else f"{self.fallback_prefix}_{idx}"
        return {
            "label": label,
            "confidence": round(float(probabilities[idx]), 4),
            "probabilities": {
                self.classes[i]: round(float(probabilities[i]), 4)
                for i in range(min(len(self.classes), len(probabilities)))
            },
        }

    def predict_with_status(
        self,
        rgb_hwc: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[dict | None, str]:
        if not self.load() or self.session is None:
            return self._default_prediction(), "classifier_unavailable_default_label"
        if mask is None or not np.any(mask > 0):
            bbox_mask = _bbox_mask(rgb_hwc.shape[:2], bbox)
            result = self.predict(rgb_hwc, bbox_mask, bbox)
            return (
                result or self._default_prediction(),
                "classified_bbox_patch_fallback" if result is not None else "empty_mask_default_label",
            )

        result = self.predict(rgb_hwc, mask, bbox)
        if result is None:
            bbox_mask = _bbox_mask(rgb_hwc.shape[:2], bbox)
            result = self.predict(rgb_hwc, bbox_mask, bbox)
            return (
                result or self._default_prediction(),
                "classified_bbox_patch_fallback" if result is not None else "empty_patch_default_label",
            )
        return result, "classified"

    def _default_prediction(self) -> dict:
        label = self.classes[0] if self.classes else self.default_label
        return {
            "label": label,
            "confidence": 0.0,
            "probabilities": {label: 1.0},
        }


_roof_classifier = MaskedPatchClassifier(
    settings.ROOF_CLASSIFIER_ONNX_PATH,
    settings.ROOF_CLASSIFIER_CLASSES_PATH,
    "roof_type",
    "roof_type_1",
)
_road_classifier = MaskedPatchClassifier(
    settings.ROAD_CLASSIFIER_ONNX_PATH,
    settings.ROAD_CLASSIFIER_CLASSES_PATH,
    "road_type",
    "road_type_3",
)


def classifier_status() -> dict:
    """Return load status for optional subtype classifiers."""
    roof_available = _roof_classifier.load()
    road_available = _road_classifier.load()
    return {
        "roof": {
            "available": roof_available,
            "model_path": settings.ROOF_CLASSIFIER_ONNX_PATH,
            "classes": _roof_classifier.classes,
        },
        "road": {
            "available": road_available,
            "model_path": settings.ROAD_CLASSIFIER_ONNX_PATH,
            "classes": _road_classifier.classes,
        },
    }


class DetectionModel:
    """Singleton wrapper around the exported SegFormer ONNX model."""

    _instance: Optional["DetectionModel"] = None

    def __init__(self):
        self.session:     Optional[ort.InferenceSession] = None
        self.input_name:  Optional[str] = None
        self.output_name: Optional[str] = None
        self.providers:   list[str] = []
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
        providers  = ["CPUExecutionProvider"]
        available  = ort.get_available_providers()
        if (
            settings.MODEL_DEVICE.lower() == "cuda"
            and "CUDAExecutionProvider" in available
        ):
            providers.insert(0, "CUDAExecutionProvider")

        logger.info("Loading SegFormer ONNX model from %s", model_path)
        self.session     = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers   = providers
        self._loaded     = True
        logger.info("ONNX model loaded with providers: %s", providers)

    def infer_chunk(self, chunk: ImageChunk) -> List[dict]:
        """
        Run SegFormer on one chunk and return per-feature detections.

        Buildings are passed through the Adaptive Building Instance Separator
        (ABIS) so each returned detection represents one individual building
        with its own mask, polygon, and roof features.

        All other feature types (road, water, …) keep the original contour-
        based detection path.

        Returns
        -------
        List[dict]
            Each dict contains at minimum:
              feature_type, confidence, chunk_id, pixel_bbox,
              geo_polygon, crs, area_px, colour.
            Building dicts additionally contain:
              building_id, roof_features (list[float]).
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

        probabilities = _softmax(logits, axis=0)
        pred_np = np.argmax(logits, axis=0).astype(np.uint8)
        return _detections_from_prediction(pred_np, probabilities, chunk, pixels)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detections_from_prediction(
    pred_np: np.ndarray,
    probabilities: np.ndarray,
    chunk: ImageChunk,
    pixels_chw: np.ndarray,
) -> list[dict]:
    """
    Convert a class mask into per-feature detections.

    Buildings are routed through BuildingSeparator; all other classes use the
    contour extraction path.
    """
    height, width = pred_np.shape
    detections: list[dict] = []

    # CHW float32 → HWC uint8 for OpenCV / BuildingSeparator
    rgb_hwc = _chw_float_to_hwc_uint8(pixels_chw)

    for feature_name, class_ids in settings.FEATURE_CLASS_MAP.items():
        if not class_ids:
            continue

        if feature_name == "building":
            semantic_mask = np.isin(pred_np, class_ids).astype(np.uint8)
            road_probability = _class_probability(
                probabilities,
                list(_feature_class_ids("road")),
            )
            mask = _suppress_building_mask_with_roads(
                semantic_mask,
                pred_np,
                road_probability,
            )
            if mask.sum() == 0:
                continue
            building_dets = _building_detections(
                rgb_hwc, mask, semantic_mask, chunk, height, width
            )
            detections.extend(building_dets)
        elif feature_name == "road" and settings.ROAD_RECONSTRUCTION_ENABLED:
            road_probability = _class_probability(probabilities, class_ids)
            model_road_mask = np.isin(pred_np, class_ids).astype(np.uint8)
            seed_mask = _road_seed_mask(pred_np, class_ids, road_probability)
            if seed_mask.sum() == 0:
                continue
            reconstructed = _road_reconstructor.reconstruct(
                seed_mask,
                road_probability,
                pred_np,
                _feature_class_ids("building"),
                _feature_class_ids("water"),
                rgb_hwc,
            )
            model_road_core = _core_only_mask(model_road_mask, chunk)
            reconstructed_core = _core_only_mask(reconstructed, chunk)
            added_core = np.logical_and(
                reconstructed_core > 0,
                model_road_core == 0,
            ).astype(np.uint8)

            if model_road_core.sum() > 0:
                detections.extend(
                    _contour_detections(
                        model_road_core,
                        feature_name,
                        chunk,
                        height,
                        width,
                        rgb_hwc,
                        model_road_mask,
                    )
                )
            if added_core.sum() > 0:
                detections.extend(
                    _contour_detections(
                        added_core,
                        "road_added",
                        chunk,
                        height,
                        width,
                        rgb_hwc,
                        model_road_mask,
                    )
                )
        else:
            mask = np.isin(pred_np, class_ids).astype(np.uint8)
            mask = _core_only_mask(mask, chunk)
            if mask.sum() == 0:
                continue
            other_dets = _contour_detections(
                mask, feature_name, chunk, height, width, rgb_hwc, mask
            )
            detections.extend(other_dets)

    return detections


def _building_detections(
    rgb_hwc: np.ndarray,
    building_mask: np.ndarray,
    semantic_building_mask: np.ndarray,
    chunk: ImageChunk,
    height: int,
    width: int,
) -> list[dict]:
    """
    Use BuildingSeparator to produce one detection per building instance.

    Each detection includes building_id and roof_features in addition to
    the standard detection fields.
    """
    colour = FEATURE_COLOURS["building"]

    try:
        inventory = _separator.extract_instances(rgb_hwc, building_mask)
    except Exception:
        logger.exception(
            "BuildingSeparator failed on chunk %s — falling back to contour path",
            chunk.chunk_id,
        )
        return _contour_detections(
            building_mask,
            "building",
            chunk,
            height,
            width,
            rgb_hwc,
            semantic_building_mask,
        )

    detections = []
    for entry in inventory:
        instance_mask: np.ndarray = entry["mask"]
        area_px = int(np.sum(instance_mask))
        if area_px < settings.MIN_SEGMENT_AREA_PX:
            continue

        x, y, w, h = entry["bbox"]
        if not _bbox_center_in_chunk_core((x, y, w, h), chunk, building_mask.shape):
            continue

        # Build geo polygon from the stored OpenCV contour
        contour = entry["polygon"]
        geo_poly = _contour_to_geo_polygon(contour, chunk.geo_transform)
        if len(geo_poly) < 4:
            continue

        confidence = round(min(float(area_px) / (width * height) * 20, 0.99), 4)

        roof_feat = entry.get("roof_features")
        roof_list = roof_feat.tolist() if roof_feat is not None else []

        classifier_mask = np.logical_and(
            instance_mask > 0,
            semantic_building_mask > 0,
        ).astype(np.uint8)
        classifier_result, classifier_status_value = _roof_classifier.predict_with_status(
            rgb_hwc,
            classifier_mask,
            (int(x), int(y), int(w), int(h)),
        )
        classifier_input = _classifier_input_from_status(
            classifier_status_value,
            "segformer_masked_patch",
        )
        if classifier_result is None:
            classifier_result, classifier_status_value = _roof_classifier.predict_with_status(
                rgb_hwc,
                instance_mask.astype(np.uint8),
                (int(x), int(y), int(w), int(h)),
            )
            classifier_input = "instance_masked_patch_fallback"
        feature_type = (
            classifier_result["label"]
            if classifier_result is not None
            else "building"
        )
        subtype = (
            classifier_result["label"]
            if classifier_result is not None
            else _subtype_from_feature_type(feature_type, "building") or _default_subtype("building")
        )
        display_label = f"building - {subtype}" if subtype else "building"

        detection = {
            "feature_type":  feature_type,
            "display_label": display_label,
            "base_feature_type": "building",
            "subtype": subtype,
            "building_id":   entry["building_id"],
            "confidence":    confidence,
            "chunk_id":      chunk.chunk_id,
            "pixel_bbox":    [int(x), int(y), int(x + w), int(y + h)],
            "geo_polygon":   geo_poly,
            "crs":           chunk.crs,
            "area_px":       area_px,
            "colour":        colour,
            "roof_features": roof_list,
        }
        if classifier_result is not None:
            detection["classifier"] = "roof"
            detection["classifier_input"] = classifier_input
            detection["classifier_status"] = classifier_status_value
            detection["classifier_confidence"] = classifier_result["confidence"]
            detection["class_probabilities"] = classifier_result["probabilities"]
        else:
            detection["classifier"] = "roof"
            detection["classifier_input"] = classifier_input
            detection["classifier_status"] = classifier_status_value
        detections.append(detection)

    logger.debug(
        "Chunk %s — BuildingSeparator produced %d building instances",
        chunk.chunk_id, len(detections),
    )
    return detections


def _contour_detections(
    mask: np.ndarray,
    feature_name: str,
    chunk: ImageChunk,
    height: int,
    width: int,
    rgb_hwc: np.ndarray | None = None,
    classifier_mask: np.ndarray | None = None,
) -> list[dict]:
    """Original contour-based detection path for non-building features."""
    colour = FEATURE_COLOURS.get(feature_name, "#888888")

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < settings.MIN_SEGMENT_AREA_PX:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon   = max(1.0, 0.002 * perimeter)
        contour   = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(contour) < 3:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        geo_poly   = _contour_to_geo_polygon(contour, chunk.geo_transform)
        if len(geo_poly) < 4:
            continue

        confidence = round(min(float(area) / (width * height) * 20, 0.99), 4)
        output_feature_name = feature_name
        detection_mask = None
        classifier_result = None
        classifier_status_value = "not_requested"
        classifier_input = "segformer_masked_patch"
        classifier_source_mask = classifier_mask if classifier_mask is not None else mask
        if feature_name == "building" and rgb_hwc is not None:
            detection_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(detection_mask, [contour], -1, 1, thickness=-1)
            detection_mask = np.logical_and(
                detection_mask > 0,
                classifier_source_mask > 0,
            ).astype(np.uint8)
            classifier_result, classifier_status_value = _roof_classifier.predict_with_status(
                rgb_hwc,
                detection_mask,
                (int(x), int(y), int(w), int(h)),
            )
            classifier_input = _classifier_input_from_status(
                classifier_status_value,
                "segformer_masked_patch",
            )
            if classifier_result is None:
                fallback_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(fallback_mask, [contour], -1, 1, thickness=-1)
                fallback_mask = np.logical_and(fallback_mask > 0, mask > 0).astype(np.uint8)
                classifier_result, classifier_status_value = _roof_classifier.predict_with_status(
                    rgb_hwc,
                    fallback_mask,
                    (int(x), int(y), int(w), int(h)),
                )
                classifier_input = "instance_masked_patch_fallback"
            if classifier_result is not None:
                output_feature_name = classifier_result["label"]
        elif feature_name in {"road", "road_added"} and rgb_hwc is not None:
            detection_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(detection_mask, [contour], -1, 1, thickness=-1)
            detection_mask = np.logical_and(
                detection_mask > 0,
                classifier_source_mask > 0,
            ).astype(np.uint8)
            classifier_result, classifier_status_value = _road_classifier.predict_with_status(
                rgb_hwc,
                detection_mask,
                (int(x), int(y), int(w), int(h)),
            )
            classifier_input = _classifier_input_from_status(
                classifier_status_value,
                "segformer_masked_patch",
            )
            if classifier_result is None:
                fallback_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(fallback_mask, [contour], -1, 1, thickness=-1)
                fallback_mask = np.logical_and(fallback_mask > 0, mask > 0).astype(np.uint8)
                classifier_result, classifier_status_value = _road_classifier.predict_with_status(
                    rgb_hwc,
                    fallback_mask,
                    (int(x), int(y), int(w), int(h)),
                )
                classifier_input = "contour_masked_patch_fallback"
            if classifier_result is not None:
                output_feature_name = classifier_result["label"]

        base_feature_type = _base_feature_type(feature_name)
        subtype = (
            classifier_result["label"]
            if classifier_result is not None
            else _subtype_from_feature_type(output_feature_name, base_feature_type)
            or _unknown_subtype(base_feature_type)
        )
        display_label = (
            f"{base_feature_type} - {subtype}"
            if subtype
            else base_feature_type
        )

        detection = {
            "feature_type": output_feature_name,
            "display_label": display_label,
            "base_feature_type": base_feature_type,
            "subtype": subtype,
            "confidence":   confidence,
            "chunk_id":     chunk.chunk_id,
            "pixel_bbox":   [int(x), int(y), int(x + w), int(y + h)],
            "geo_polygon":  geo_poly,
            "crs":          chunk.crs,
            "area_px":      int(area),
            "colour":       colour,
        }
        if feature_name == "road_added":
            detection["source_feature_type"] = "road_added"
        if classifier_result is not None:
            detection["classifier"] = "roof" if feature_name == "building" else "road"
            detection["classifier_input"] = classifier_input
            detection["classifier_status"] = classifier_status_value
            detection["classifier_confidence"] = classifier_result["confidence"]
            detection["class_probabilities"] = classifier_result["probabilities"]
        elif feature_name in {"building", "road", "road_added"}:
            detection["classifier"] = "roof" if feature_name == "building" else "road"
            detection["classifier_input"] = classifier_input
            detection["classifier_status"] = classifier_status_value
        detections.append(detection)

    return detections


def _chw_float_to_hwc_uint8(pixels_chw: np.ndarray) -> np.ndarray:
    """Convert a CHW float32 [0,1] array to an HWC uint8 [0,255] array."""
    hwc = np.transpose(pixels_chw[:3], (1, 2, 0))
    return (np.clip(hwc, 0.0, 1.0) * 255).astype(np.uint8)


def _masked_rgb_patch(
    rgb_hwc: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(rgb_hwc.shape[1], x0 + int(w))
    y1 = min(rgb_hwc.shape[0], y0 + int(h))
    if x0 >= x1 or y0 >= y1:
        return None

    patch = rgb_hwc[y0:y1, x0:x1].copy()
    patch_mask = (mask[y0:y1, x0:x1] > 0)
    if not np.any(patch_mask):
        return None

    patch[~patch_mask] = 0
    return patch


def _bbox_mask(
    shape_hw: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = shape_hw
    x, y, w, h = bbox
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, x0 + max(0, int(w)))
    y1 = min(height, y0 + max(0, int(h)))
    mask = np.zeros((height, width), dtype=np.uint8)
    if x0 < x1 and y0 < y1:
        mask[y0:y1, x0:x1] = 1
    return mask


def _classifier_tensor(
    patch_rgb: np.ndarray,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    resized = cv2.resize(
        patch_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    resized /= 255.0
    resized = (resized - mean.reshape(1, 3)) / std.reshape(1, 3)
    chw = np.transpose(resized, (2, 0, 1))
    return chw[np.newaxis, ...].astype(np.float32, copy=False)


def _base_feature_type(feature_name: str) -> str:
    if feature_name in {"road", "road_added"}:
        return "road"
    if feature_name.startswith("road_type_"):
        return "road"
    if feature_name.startswith("roof_type_"):
        return "building"
    return feature_name


def _subtype_from_feature_type(feature_name: str, base_feature_type: str) -> str | None:
    if feature_name.startswith(("roof_type_", "road_type_")):
        return feature_name
    if feature_name != base_feature_type:
        return feature_name
    return None


def _unknown_subtype(base_feature_type: str) -> str | None:
    return _default_subtype(base_feature_type)


def _default_subtype(base_feature_type: str) -> str | None:
    if base_feature_type == "building":
        return _roof_classifier.classes[0] if _roof_classifier.classes else "roof_type_1"
    if base_feature_type == "road":
        return _road_classifier.classes[0] if _road_classifier.classes else "road_type_3"
    return None


def _classifier_input_from_status(status: str, default_input: str) -> str:
    if status == "classified_bbox_patch_fallback":
        return "bbox_patch_fallback"
    if status.endswith("_default_label"):
        return "default_label"
    return default_input


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
    height, width = size_hw
    resized = [
        cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
        for channel in array
    ]
    return np.stack(resized, axis=0).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _class_probability(probabilities: np.ndarray, class_ids: list[int]) -> np.ndarray:
    valid_ids = [class_id for class_id in class_ids if class_id < probabilities.shape[0]]
    if not valid_ids:
        return np.zeros(probabilities.shape[1:], dtype=np.float32)
    return np.max(probabilities[valid_ids], axis=0).astype(np.float32, copy=False)


def _feature_class_ids(feature_name: str) -> set[int]:
    return set(int(class_id) for class_id in settings.FEATURE_CLASS_MAP.get(feature_name, []))


def _suppress_building_mask_with_roads(
    building_mask: np.ndarray,
    pred_np: np.ndarray,
    road_probability: np.ndarray,
) -> np.ndarray:
    """Give SegFormer road evidence priority over building pixels."""
    road_ids = _feature_class_ids("road")
    if not road_ids:
        return building_mask

    road_conflict = np.isin(pred_np, list(road_ids))
    road_conflict |= road_probability >= settings.ROAD_BUILDING_SUPPRESS_PROB

    dilate_px = int(settings.ROAD_BUILDING_SUPPRESS_DILATE)
    if dilate_px > 0:
        kernel_size = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        road_conflict = cv2.dilate(
            road_conflict.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)

    cleaned = building_mask.copy()
    cleaned[road_conflict] = 0
    return cleaned.astype(np.uint8)


def _road_seed_mask(
    pred_np: np.ndarray,
    class_ids: list[int],
    road_probability: np.ndarray,
) -> np.ndarray:
    """Seed roads from both argmax road pixels and high road probability pixels."""
    mask = np.isin(pred_np, class_ids)
    mask |= road_probability >= settings.ROAD_SEED_PROBABILITY

    dilate_px = int(settings.ROAD_PROBABILITY_DILATE)
    if dilate_px > 0:
        kernel_size = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    return mask.astype(np.uint8)


def _contour_to_geo_polygon(contour: np.ndarray | list, geo_transform: list) -> list:
    """Convert an OpenCV contour or point list to a closed polygon in source CRS."""
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    polygon = []
    for point in points:
        col, row = int(point[0]), int(point[1])
        polygon.append(_px_to_geo(col, row, geo_transform))
    if polygon and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon


def _core_only_mask(mask: np.ndarray, chunk: ImageChunk) -> np.ndarray:
    """Keep only the non-overlap tile core to avoid duplicate detections."""
    core_bounds = _chunk_core_bounds(mask.shape, chunk)
    if core_bounds is None:
        return np.zeros_like(mask)

    x0, y0, x1, y1 = core_bounds
    core = np.zeros_like(mask)
    core[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return core


def _bbox_center_in_chunk_core(
    bbox: tuple[int, int, int, int],
    chunk: ImageChunk,
    mask_shape: tuple[int, int],
) -> bool:
    """Return true when a bbox center falls inside this chunk's non-overlap core."""
    core_bounds = _chunk_core_bounds(mask_shape, chunk)
    if core_bounds is None:
        return False

    x, y, w, h = bbox
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    x0, y0, x1, y1 = core_bounds
    return x0 <= cx < x1 and y0 <= cy < y1


def _chunk_core_bounds(
    mask_shape: tuple[int, int],
    chunk: ImageChunk,
) -> tuple[int, int, int, int] | None:
    """Return core bounds as x0, y0, x1, y1 in chunk-local pixels."""
    chunk_size   = settings.CHUNK_SIZE
    core_col_off = chunk.col * chunk_size
    core_row_off = chunk.row * chunk_size
    core_width   = min(chunk_size, chunk.image_width  - core_col_off)
    core_height  = min(chunk_size, chunk.image_height - core_row_off)
    if core_width <= 0 or core_height <= 0:
        return None

    x0 = int(round(core_col_off - chunk.window.col_off))
    y0 = int(round(core_row_off - chunk.window.row_off))
    x1 = min(mask_shape[1], x0 + core_width)
    y1 = min(mask_shape[0], y0 + core_height)
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x0 >= x1 or y0 >= y1:
        return None

    return x0, y0, x1, y1


def _px_to_geo(col: int, row: int, geo_transform: list) -> list:
    """Convert one pixel coordinate to source CRS coordinates."""
    x = geo_transform[0] + col * geo_transform[1] + row * geo_transform[2]
    y = geo_transform[3] + col * geo_transform[4] + row * geo_transform[5]
    return [round(x, 8), round(y, 8)]


def _resolve_model_path(model_path: str) -> Path:
    path       = Path(model_path)
    candidates = [path, Path.cwd() / path]

    service_path = Path(__file__).resolve()
    for parent in service_path.parents:
        candidates.append(parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(c) for c in candidates[:5])
    raise FileNotFoundError(
        f"ONNX model not found: {model_path}. Searched: {searched}"
    )


detection_model = DetectionModel.get()
