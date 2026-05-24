"""Post-processing helpers for streamed segmentation detections."""

from collections import defaultdict
from typing import Iterable

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def merge_same_label_detections(detections: Iterable[dict]) -> list[dict]:
    """
    Dissolve touching or overlapping polygons that have the same label and CRS.

    Live inference still streams per-chunk polygons so the UI can update quickly.
    Once a job completes, this function creates cleaner final segmentation
    polygons without duplicate overlap boundaries between adjacent chunks.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    passthrough = []

    for detection in detections:
        polygon = detection.get("geo_polygon") or []
        if len(polygon) < 4:
            passthrough.append(detection)
            continue
        key = (
            detection.get("feature_type", "unknown"),
            detection.get("crs", "EPSG:4326"),
        )
        groups[key].append(detection)

    merged = list(passthrough)
    for (feature_type, crs), group in groups.items():
        shapes = []
        total_area_px = 0
        max_confidence = 0.0
        colour = group[0].get("colour", "#888888")

        for detection in group:
            polygon = _safe_polygon(detection.get("geo_polygon") or [])
            if polygon is None:
                continue
            shapes.append(polygon)
            total_area_px += int(detection.get("area_px") or 0)
            max_confidence = max(max_confidence, float(detection.get("confidence") or 0))

        if not shapes:
            continue

        dissolved = unary_union(shapes)
        polygons = [polygon for polygon in _iter_polygons(dissolved) if not polygon.is_empty]
        total_geometry_area = sum(polygon.area for polygon in polygons)

        for idx, polygon in enumerate(polygons):
            if polygon.is_empty:
                continue
            ring = [
                [round(float(x), 8), round(float(y), 8)]
                for x, y in polygon.exterior.coords
            ]
            if len(ring) < 4:
                continue
            area_px = total_area_px
            if total_geometry_area > 0:
                area_px = round(total_area_px * (polygon.area / total_geometry_area))
            merged.append({
                "feature_type": feature_type,
                "confidence": round(max_confidence, 4),
                "chunk_id": f"merged_{feature_type}_{idx}",
                "pixel_bbox": None,
                "geo_polygon": ring,
                "crs": crs,
                "area_px": int(area_px),
                "colour": colour,
            })

    return merged


def _safe_polygon(points: list) -> Polygon | None:
    try:
        polygon = Polygon(points)
    except Exception:
        return None

    if polygon.is_empty or polygon.area <= 0:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    return polygon


def _iter_polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
