"""
Full-image road network post-processing.

Runs AFTER all per-chunk SegFormer inference AND the initial Shapely polygon
merge.  By working on the complete merged road detection set it can fix gaps
that span chunk boundaries — something per-chunk processing cannot do.

Pipeline
--------
1. Rasterize all merged road polygons → downscaled binary mask
2. Large morphological closing         → bridges cross-chunk & shadow gaps
3. Template matching at 8 road angles  → detects aligned gap patterns
4. Skeleton endpoint linking           → connects broken stubs precisely
5. Final cleanup                       → remove noise, re-apply obstacles
6. Re-vectorize                        → GeoJSON polygons from enhanced mask

Memory: the working mask targets ≤ 2000 px on the longest axis,
so even a 6 GB GeoTIFF produces a ~2 MB uint8 mask.
"""

import logging
import math
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import rasterio

from app.core.config import settings

logger = logging.getLogger(__name__)

_MASK_MAX_DIM = 2000   # longest side of the downscaled working mask

# Ordered list of enhancement step names (used by the progress callback)
ENHANCEMENT_STEPS = [
    "Building Road Mask",
    "Running Template Matching",
    "Detecting Broken Roads",
    "Running Connected Component Analysis",
    "Finding Road Endpoints",
    "Matching Road Segments",
    "Connecting Roads",
    "Removing Outliers",
    "Smoothing Network",
    "Updating Visualization",
]
_TOTAL_STEPS = len(ENHANCEMENT_STEPS)


# ── Public entry point ────────────────────────────────────────────────────────

def enhance_road_network(
    detections: List[dict],
    tif_path: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[dict]:
    """
    Enhance road connectivity on the full merged detection set.

    Parameters
    ----------
    detections        : all merged detections (buildings + roads + water)
    tif_path          : path to the source GeoTIFF (for geo-transform / dimensions)
    progress_callback : optional callable(step_name, step_index, total_steps)
                        called at each pipeline stage for UI progress reporting

    Returns all detections with road polygons replaced by enhanced versions.
    Buildings and water are returned unchanged.
    """
    _prog = progress_callback if progress_callback else (lambda *_: None)

    road_dets   = [d for d in detections if d.get("feature_type") == "road"]
    bld_dets    = [d for d in detections if d.get("feature_type") == "building"]
    water_dets  = [d for d in detections if d.get("feature_type") == "water"]
    other_dets  = [d for d in detections
                   if d.get("feature_type") not in ("road", "building", "water")]

    if not road_dets:
        logger.info("No road detections to enhance.")
        return detections

    # Open TIF for image metadata only (no pixel reads)
    with rasterio.open(tif_path) as src:
        img_w     = src.width
        img_h     = src.height
        transform = list(src.transform)[:6]
        crs       = src.crs.to_string() if src.crs else "EPSG:4326"

    scale  = max(1, max(img_w, img_h) // _MASK_MAX_DIM)
    mask_w = math.ceil(img_w / scale)
    mask_h = math.ceil(img_h / scale)

    logger.info(
        "Road network post-processing: %dx%d → 1:%d scale → %dx%d mask",
        img_w, img_h, scale, mask_w, mask_h,
    )

    _prog("Building Road Mask", 1, _TOTAL_STEPS)
    road_mask  = _rasterize(road_dets,  mask_w, mask_h, transform, scale)
    bld_mask   = _rasterize(bld_dets,   mask_w, mask_h, transform, scale)
    water_mask = _rasterize(water_dets, mask_w, mask_h, transform, scale)

    if road_mask.sum() == 0:
        return detections

    obstacle = cv2.bitwise_or(
        (bld_mask > 0).astype(np.uint8),
        (water_mask > 0).astype(np.uint8),
    )

    enhanced = _run_pipeline(road_mask, obstacle, scale, _prog)

    _prog("Updating Visualization", 10, _TOTAL_STEPS)
    new_road_dets = _vectorize(enhanced, transform, scale, crs)
    logger.info(
        "Road enhancement: %d polygons → %d polygons",
        len(road_dets), len(new_road_dets),
    )

    return bld_dets + new_road_dets + water_dets + other_dets


# ── Enhancement pipeline ──────────────────────────────────────────────────────

def _run_pipeline(
    road_mask: np.ndarray,
    obstacle:  np.ndarray,
    scale:     int,
    prog:      Callable = None,
) -> np.ndarray:
    """Run all enhancement stages on the downscaled binary road mask."""
    _p = prog if prog else (lambda *_: None)

    # ── Stage 1: large morphological closing ─────────────────────────────────
    # Bridges gaps left by tree shadows, chunk-boundary seams, and small
    # segmentation holes.  Kernel radius ≈ 20 px at downscale maps to
    # ~60-80 real pixels — enough to cover most inter-chunk gap widths.
    _p("Detecting Broken Roads", 3, _TOTAL_STEPS)
    logger.info("    Detecting broken road segments...")
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed  = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    closed[obstacle > 0] = 0

    # ── Stage 2: template matching ────────────────────────────────────────────
    # Slides dumbbell-shaped templates (two road stubs with a gap in the
    # middle) at 8 orientations across the mask.  Anywhere the pattern
    # matches above ROAD_TEMPLATE_MATCH_THRESHOLD, fill the central gap.
    _p("Running Template Matching", 2, _TOTAL_STEPS)
    logger.info("    Performing template gap fill...")
    template_out = _template_gap_fill(closed, obstacle)

    # ── Stage 3: skeleton endpoint linking ───────────────────────────────────
    # After template filling, compute a skeleton and connect any remaining
    # endpoint pairs within 2× the configured max connection distance.
    _p("Running Connected Component Analysis", 4, _TOTAL_STEPS)
    _p("Finding Road Endpoints", 5, _TOTAL_STEPS)
    logger.info("    Computing skeleton and endpoints...")
    skeleton = _compute_skeleton(template_out)

    _p("Matching Road Segments", 6, _TOTAL_STEPS)
    _p("Connecting Roads", 7, _TOTAL_STEPS)
    logger.info("    Linking road endpoints...")
    linked   = _link_endpoints(template_out.copy(), skeleton, obstacle)
    linked[obstacle > 0] = 0

    # ── Stage 4: final cleanup ────────────────────────────────────────────────
    _p("Removing Outliers", 8, _TOTAL_STEPS)
    _p("Smoothing Network", 9, _TOTAL_STEPS)
    logger.info("    Refining road network...")
    refined = _refine(linked, obstacle, scale)

    return refined


# ── Stage 2: Template matching ────────────────────────────────────────────────

def _template_gap_fill(mask: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """
    For each of 8 road orientations, build a dumbbell template
    (road pixels on both sides, empty in the centre) and use
    cv2.matchTemplate to locate gap regions.  Fill confirmed gaps.
    """
    enhanced  = mask.copy()
    threshold = settings.ROAD_TEMPLATE_MATCH_THRESHOLD
    h, w      = mask.shape

    # Template arm length in pixels (adaptive to mask size)
    arm = max(6, min(30, min(h, w) // 30))
    gap = max(2, arm // 3)            # gap width at centre
    tpl_size = 2 * arm + 2 * gap + 4  # total template side length

    angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

    for angle_deg in angles:
        rad = math.radians(angle_deg)
        dx  = math.cos(rad)
        dy  = math.sin(rad)
        cx  = cy = tpl_size // 2

        # Build dumbbell template
        tpl = np.zeros((tpl_size, tpl_size), dtype=np.float32)
        for s in range(gap, arm + gap + 1):
            for side in (-1, 1):
                r = int(round(cy + side * dy * s))
                c = int(round(cx + side * dx * s))
                if 0 <= r < tpl_size and 0 <= c < tpl_size:
                    tpl[r, c] = 1.0

        if tpl.sum() < 4:
            continue

        # Normalise so TM_CCOEFF_NORMED scores are comparable across angles
        tpl /= (tpl.sum() + 1e-6)

        road_f = mask.astype(np.float32)
        if road_f.shape[0] < tpl_size or road_f.shape[1] < tpl_size:
            continue

        result = cv2.matchTemplate(road_f, tpl, cv2.TM_CCOEFF_NORMED)
        match_rows, match_cols = np.nonzero(result >= threshold)

        for mr, mc in zip(match_rows, match_cols):
            # Centre of the matched region in the original mask
            cr = mr + cy
            cc = mc + cx
            if not (0 <= cr < h and 0 <= cc < w):
                continue

            # Draw through-line from arm-end to arm-end, filling the gap
            e1r = int(round(cr - dy * (arm + gap)))
            e1c = int(round(cc - dx * (arm + gap)))
            e2r = int(round(cr + dy * (arm + gap)))
            e2c = int(round(cc + dx * (arm + gap)))

            if _path_blocked(cr, cc, e1r, e1c, obstacle):
                continue
            if _path_blocked(cr, cc, e2r, e2c, obstacle):
                continue

            cv2.line(enhanced, (e1c, e1r), (e2c, e2r), 1, thickness=3)

    return enhanced


# ── Stage 1 helper: morphological skeleton ────────────────────────────────────

def _compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """Iterative-erosion skeleton (no opencv-contrib needed)."""
    if mask.sum() == 0:
        return np.zeros_like(mask)

    skel    = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img     = mask.copy()

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp   = cv2.subtract(img, opened)
        skel   = cv2.bitwise_or(skel, temp)   # add BEFORE emptiness check
        img    = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


# ── Stage 3: Endpoint linking ─────────────────────────────────────────────────

def _link_endpoints(
    mask:     np.ndarray,
    skeleton: np.ndarray,
    obstacle: np.ndarray,
) -> np.ndarray:
    """
    Find skeleton endpoints (pixels with ≤1 skeleton neighbour) and connect
    pairs that are within 2× ROAD_MAX_CONNECTION_DISTANCE and not
    separated by an obstacle.
    """
    if skeleton.sum() == 0:
        return mask

    # Endpoint detection
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0.0
    nbr_sum  = cv2.filter2D(skeleton.astype(np.float32), -1, kernel)
    ep_mask  = (skeleton > 0) & (nbr_sum.astype(np.uint8) <= 1)
    yx       = np.column_stack(np.nonzero(ep_mask))   # (N, 2)

    if len(yx) < 2:
        return mask

    # Cap at 400 endpoints for O(n²) performance
    if len(yx) > 400:
        step = len(yx) // 400
        yx   = yx[::step]

    max_dist = float(settings.ROAD_MAX_CONNECTION_DISTANCE) * 2.0

    # Pairwise distances via numpy broadcasting
    yx_f = yx.astype(np.float32)
    diff = yx_f[:, np.newaxis, :] - yx_f[np.newaxis, :, :]   # (N, N, 2)
    dists = np.sqrt((diff ** 2).sum(axis=2))                   # (N, N)

    i_idx, j_idx = np.nonzero((dists > 0) & (dists <= max_dist))
    # Only process each pair once
    keep = i_idx < j_idx
    i_idx, j_idx = i_idx[keep], j_idx[keep]

    for i, j in zip(i_idx, j_idx):
        r1, c1 = int(yx[i][0]), int(yx[i][1])
        r2, c2 = int(yx[j][0]), int(yx[j][1])
        if not _path_blocked(r1, c1, r2, c2, obstacle):
            cv2.line(mask, (c1, r1), (c2, r2), 1, thickness=3)

    return mask


# ── Stage 4: Cleanup ──────────────────────────────────────────────────────────

def _refine(mask: np.ndarray, obstacle: np.ndarray, scale: int) -> np.ndarray:
    """Remove tiny fragments, smooth boundaries, re-apply obstacle exclusion."""
    min_area = max(5, settings.ROAD_MIN_COMPONENT_AREA // (scale * scale))

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined, connectivity=8
    )
    clean = np.zeros_like(refined)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lbl] = 1

    if obstacle.any():
        clean[obstacle > 0] = 0

    return clean.astype(np.uint8)


# ── Rasterization ─────────────────────────────────────────────────────────────

def _rasterize(
    detections: List[dict],
    mask_w:     int,
    mask_h:     int,
    transform:  list,
    scale:      int,
) -> np.ndarray:
    """
    Draw all detection geo-polygons onto a binary uint8 mask.

    transform = [x_origin, x_res, x_rot, y_origin, y_rot, y_res]
    col = (x − x_origin) / x_res / scale
    row = (y − y_origin) / y_res / scale   (y_res is negative for north-up)
    """
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

    x_origin, x_res = transform[0], transform[1]
    y_origin, y_res = transform[3], transform[5]

    for det in detections:
        polygon = det.get("geo_polygon") or []
        if len(polygon) < 3:
            continue

        pts = []
        for pt in polygon:
            col = int((pt[0] - x_origin) / x_res / scale)
            row = int((pt[1] - y_origin) / y_res / scale)
            col = max(0, min(mask_w - 1, col))
            row = max(0, min(mask_h - 1, row))
            pts.append([col, row])

        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], 1)

    return mask


# ── Vectorization ─────────────────────────────────────────────────────────────

def _vectorize(
    mask:      np.ndarray,
    transform: list,
    scale:     int,
    crs:       str,
) -> List[dict]:
    """
    Extract contours from the enhanced binary mask and convert them to
    GeoJSON-compatible detection dicts in the source CRS.
    """
    min_area = max(5, settings.ROAD_MIN_COMPONENT_AREA // (scale * scale))

    x_origin, x_res = transform[0], transform[1]
    y_origin, y_res = transform[3], transform[5]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections  = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perim   = cv2.arcLength(contour, closed=True)
        epsilon = max(1.0, 0.002 * perim)
        contour = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(contour) < 3:
            continue

        geo_poly = []
        for pt in contour.reshape(-1, 2):
            col, row = int(pt[0]), int(pt[1])
            x = round(x_origin + col * scale * x_res, 8)
            y = round(y_origin + row * scale * y_res, 8)
            geo_poly.append([x, y])

        if len(geo_poly) < 4:
            continue

        # Close polygon
        if geo_poly[0] != geo_poly[-1]:
            geo_poly.append(geo_poly[0])

        area_px    = int(area) * scale * scale
        confidence = round(min(0.99, area_px / 1e6), 4)

        detections.append({
            "feature_type": "road",
            "confidence":   confidence,
            "chunk_id":     "road_network_enhanced",
            "pixel_bbox":   None,
            "geo_polygon":  geo_poly,
            "crs":          crs,
            "area_px":      area_px,
            "colour":       "#4488FF",
        })

    return detections


# ── Shared utility ────────────────────────────────────────────────────────────

def _path_blocked(
    r1: int, c1: int,
    r2: int, c2: int,
    obstacle: np.ndarray,
) -> bool:
    """Bresenham line check — True if any sample point is an obstacle pixel."""
    steps = max(abs(r2 - r1), abs(c2 - c1), 1)
    h, w  = obstacle.shape
    for t in range(1, steps):
        r = int(round(r1 + (r2 - r1) * t / steps))
        c = int(round(c1 + (c2 - c1) * t / steps))
        if 0 <= r < h and 0 <= c < w and obstacle[r, c]:
            return True
    return False
