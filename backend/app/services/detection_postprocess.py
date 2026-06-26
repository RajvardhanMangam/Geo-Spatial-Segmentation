"""Post-processing helpers for streamed segmentation detections."""

from collections import defaultdict
from typing import Iterable

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


# Area-overlap ratio, measured against the smaller polygon, required to treat
# two building detections as duplicates. Side-by-side buildings have zero
# intersection area and are kept separate.
_BUILDING_DUPLICATE_OVERLAP_RATIO = 0.10


def merge_same_label_detections(detections: Iterable[dict]) -> list[dict]:
    """
    Dissolve touching or overlapping polygons that share the same label and CRS.

    Buildings are **not** re-merged here — they have already been separated into
    individual instances by BuildingSeparator (ABIS) and must remain distinct.
    All non-building feature types (road, water, …) are dissolved as before.

    Live inference streams per-chunk polygons so the UI updates quickly.
    This function is called once after all chunks complete to produce clean
    final geometries without duplicate overlap boundaries.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    passthrough: list[dict] = []
    building_detections: list[dict] = []

    for detection in detections:
        polygon = detection.get("geo_polygon") or []
        feature_type = detection.get("feature_type", "unknown")
        base_feature_type = detection.get("base_feature_type", feature_type)

        # Buildings keep their individual identity — never dissolve them.
        if base_feature_type == "building":
            building_detections.append(detection)
            continue

        if len(polygon) < 4:
            passthrough.append(detection)
            continue

        key = (feature_type, detection.get("crs", "EPSG:4326"))
        groups[key].append(detection)

    merged = list(passthrough)
    merged.extend(_dedupe_building_detections(building_detections))

    for (feature_type, crs), group in groups.items():
        shapes:         list[Polygon] = []
        total_area_px:  int   = 0
        max_confidence: float = 0.0
        colour = group[0].get("colour", "#888888")
        base_feature_type = group[0].get("base_feature_type", feature_type)
        subtype = group[0].get("subtype") or _subtype_from_feature_type(
            feature_type,
            base_feature_type,
        ) or _unknown_subtype(base_feature_type)
        classifier = group[0].get("classifier")
        classifier_confidence = max(
            float(detection.get("classifier_confidence") or 0)
            for detection in group
        )

        for detection in group:
            polygon = _safe_polygon(detection.get("geo_polygon") or [])
            if polygon is None:
                continue
            shapes.append(polygon)
            total_area_px  += int(detection.get("area_px") or 0)
            max_confidence  = max(max_confidence, float(detection.get("confidence") or 0))

        if not shapes:
            continue

        dissolved = unary_union(shapes)
        polygons  = [p for p in _iter_polygons(dissolved) if not p.is_empty]
        total_geo_area = sum(p.area for p in polygons)

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
            if total_geo_area > 0:
                area_px = round(total_area_px * (polygon.area / total_geo_area))

            merged_detection = {
                "feature_type": feature_type,
                "display_label": group[0].get("display_label", feature_type),
                "base_feature_type": base_feature_type,
                "subtype": subtype,
                "confidence":   round(max_confidence, 4),
                "chunk_id":     f"merged_{feature_type}_{idx}",
                "pixel_bbox":   None,
                "geo_polygon":  ring,
                "crs":          crs,
                "area_px":      int(area_px),
                "colour":       colour,
            }
            if classifier:
                merged_detection["classifier"] = classifier
                merged_detection["classifier_confidence"] = round(classifier_confidence, 4)
            if any(detection.get("source_feature_type") == "road_added" for detection in group):
                merged_detection["source_feature_type"] = "road_added"
            merged.append(merged_detection)

    return merged


def assign_global_building_ids(detections: list[dict]) -> list[dict]:
    """
    Re-number building detections with globally unique, zero-padded IDs.

    BuildingSeparator assigns IDs that are local to each chunk (1, 2, 3, …).
    After all chunks are collected this function replaces those with IDs that
    are unique across the whole image (B000001, B000002, …).

    Non-building detections are returned unchanged.

    Parameters
    ----------
    detections : list[dict]
        All detections for a completed job, as stored in Redis.

    Returns
    -------
    list[dict]
        Same list with building ``building_id`` fields replaced.
    """
    result:   list[dict] = []
    counter:  int        = 1

    for det in detections:
        if det.get("base_feature_type", det.get("feature_type")) == "building":
            det = {**det, "building_id": f"B{counter:06d}"}
            counter += 1
        result.append(det)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _dedupe_building_detections(detections: list[dict]) -> list[dict]:
    """Drop duplicate building detections that overlap in area."""
    valid: list[tuple[dict, Polygon]] = []
    passthrough: list[dict] = []

    for detection in detections:
        polygon = _safe_polygon(detection.get("geo_polygon") or [])
        if polygon is None:
            passthrough.append(detection)
            continue
        valid.append((detection, polygon))

    valid.sort(
        key=lambda item: (
            float(item[0].get("confidence") or 0),
            float(item[0].get("area_px") or 0),
            item[1].area,
        ),
        reverse=True,
    )

    kept: list[tuple[dict, Polygon]] = []
    for detection, polygon in valid:
        duplicate = False
        for _, kept_polygon in kept:
            intersection_area = polygon.intersection(kept_polygon).area
            if intersection_area <= 0:
                continue
            smaller_area = min(polygon.area, kept_polygon.area)
            if smaller_area <= 0:
                continue
            if intersection_area / smaller_area >= _BUILDING_DUPLICATE_OVERLAP_RATIO:
                duplicate = True
                break

        if not duplicate:
            kept.append((detection, polygon))

    return passthrough + [detection for detection, _ in kept]


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


def _subtype_from_feature_type(feature_type: str, base_feature_type: str) -> str | None:
    if feature_type.startswith(("roof_type_", "road_type_")):
        return feature_type
    if feature_type != base_feature_type:
        return feature_type
    return None


def _unknown_subtype(base_feature_type: str) -> str | None:
    if base_feature_type == "building":
        return "roof_type_1"
    if base_feature_type == "road":
        return "road_type_3"
    return None


def _iter_polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
