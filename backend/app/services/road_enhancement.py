"""
Road connectivity enhancement — additional post-processing stage.

Applied AFTER SegFormer generates a road mask and AFTER _core_only_mask crops
it to the non-overlap tile, but BEFORE contour extraction.

Buildings and water are never modified by this module.

Five stages
-----------
1. Road gap detection   – skeleton + endpoint analysis
2. Component analysis   – CC stats, orientation, width
3. Intelligent linking  – connect compatible endpoint pairs
4. Network refinement   – gap fill, smooth, remove artifacts
5. Chunk-edge continuity – directional dilation to tile border so Shapely merge joins them

Log analysis revealed most chunks are "single component", meaning gaps are
between chunks (not within them).  Stage 5 is therefore the highest-impact
stage and uses directional morphological dilation rather than skeleton
endpoints for reliable coverage up to the full chunk-overlap distance.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# Pixels from the core boundary that are scanned for boundary extension.
# Set to the full chunk overlap so any road that the model predicted inside
# the overlap zone is guaranteed to be pushed to the tile edge.
_EDGE_MARGIN = 64


# ── Public entry point ─────────────────────────────────────────────────────────

def enhance_road_mask(
    road_mask: np.ndarray,
    building_mask: Optional[np.ndarray] = None,
    water_mask: Optional[np.ndarray] = None,
    chunk=None,
) -> np.ndarray:
    """
    Enhance road connectivity in a binary uint8 road mask.

    Parameters
    ----------
    road_mask      : uint8 ndarray (1=road, 0=background), shape (H, W).
                     Must already have _core_only_mask applied.
    building_mask  : uint8 ndarray, same shape – blocks road bridges
    water_mask     : uint8 ndarray, same shape – blocks road bridges
    chunk          : ImageChunk, used for Stage 5 boundary detection

    Returns a refined road mask of the same shape and dtype.
    All other feature types (buildings, water) are completely unchanged.
    """
    if not settings.ROAD_CONNECTIVITY_ENABLE:
        return road_mask

    if road_mask.sum() == 0:
        return road_mask

    logger.info("  Enhancing road connectivity...")

    obstacle = _build_obstacle_mask(road_mask, building_mask, water_mask)

    # Pre-step: morphological closing to merge fragments separated by small gaps
    # before component analysis.  This reduces false "single component" results
    # caused by tiny breaks in an otherwise continuous road prediction.
    pre_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    pre_closed = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, pre_k, iterations=1)
    pre_closed[obstacle > 0] = 0

    # Stage 1 – detect broken segments via skeleton on the pre-closed mask
    logger.info("    Detecting broken road segments...")
    skeleton = _compute_skeleton(pre_closed)

    # Stage 2 – connected component analysis on the pre-closed mask
    logger.info("    Performing connected component analysis...")
    components = _analyze_components(pre_closed, skeleton)

    if len(components) < 2:
        # Single (or no) component: skip linking, go straight to refinement
        logger.info("    Performing connected component analysis...")
        logger.info("    Finding candidate road endpoints...")
        logger.info("    Matching road fragments...")
        logger.info("    Connecting compatible road segments...")
        logger.info("    Refining road network...")
        enhanced = _refine_mask(pre_closed, obstacle)
    else:
        # Stage 3 – intelligent linking across multiple components
        logger.info("    Finding candidate road endpoints...")
        logger.info("    Matching road fragments...")
        logger.info("    Connecting compatible road segments...")
        enhanced = _link_road_segments(pre_closed.copy(), components, obstacle)

        # Stage 4 – network refinement
        logger.info("    Refining road network...")
        enhanced = _refine_mask(enhanced, obstacle)

    # Stage 5 – chunk-edge continuity (highest-impact for cross-chunk gaps)
    if settings.ROAD_CHUNK_EDGE_CONNECTION_ENABLE and chunk is not None:
        enhanced = _handle_chunk_boundary(enhanced, obstacle, chunk)

    logger.info("  Road enhancement completed.")
    return enhanced


# ── Stage 1: Skeleton ──────────────────────────────────────────────────────────

def _compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Return a morphological skeleton of the road mask via iterative erosion.
    No opencv-contrib required.
    """
    if not settings.ROAD_SKELETON_ENABLE or mask.sum() == 0:
        return np.zeros_like(mask)

    skel = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)   # add BEFORE checking if done
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


def _skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Boolean mask of skeleton pixels with exactly 1 neighbour (tips/endpoints)."""
    if skeleton.sum() == 0:
        return np.zeros(skeleton.shape, dtype=bool)

    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0.0
    neighbour_sum = cv2.filter2D(skeleton.astype(np.float32), -1, kernel)
    return (skeleton > 0) & (neighbour_sum.astype(np.uint8) == 1)


# ── Stage 2: Connected component analysis ─────────────────────────────────────

@dataclass
class _RoadComponent:
    label: int
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    orientation: float
    endpoints: List[Tuple[int, int]]
    avg_width: float


def _analyze_components(
    road_mask: np.ndarray,
    skeleton: np.ndarray,
) -> List[_RoadComponent]:
    """Compute properties for every connected component above MIN_COMPONENT_AREA."""
    min_area = settings.ROAD_MIN_COMPONENT_AREA
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        road_mask, connectivity=8
    )

    ep_mask = _skeleton_endpoints(skeleton)
    components: List[_RoadComponent] = []

    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[lbl][0])
        cy = float(centroids[lbl][1])

        comp_binary = (labels == lbl).astype(np.uint8)
        orientation = _pca_orientation(comp_binary)
        endpoints = _component_endpoints(comp_binary, ep_mask)
        avg_width = _average_width(comp_binary, skeleton)

        components.append(_RoadComponent(
            label=lbl,
            area=area,
            bbox=(x, y, w, h),
            centroid=(cx, cy),
            orientation=orientation,
            endpoints=endpoints,
            avg_width=avg_width,
        ))

    return components


def _pca_orientation(comp_mask: np.ndarray) -> float:
    """Principal orientation via PCA on pixel coordinates, in degrees [0, 180)."""
    yx = np.column_stack(np.nonzero(comp_mask)).astype(np.float32)
    if len(yx) < 2:
        return 0.0
    mean = yx.mean(axis=0)
    cov = np.cov((yx - mean).T)
    if cov.ndim < 2:
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    return math.degrees(math.atan2(float(principal[1]), float(principal[0]))) % 180.0


def _component_endpoints(
    comp_mask: np.ndarray,
    ep_mask: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Skeleton endpoints belonging to this component, with PCA-extremal fallback.
    """
    combined = comp_mask.astype(bool) & ep_mask
    pts = list(zip(*np.nonzero(combined)))

    if not pts:
        yx = np.column_stack(np.nonzero(comp_mask))
        if len(yx) < 2:
            return [(int(yx[0][0]), int(yx[0][1]))] if len(yx) == 1 else []
        mean = yx.mean(axis=0)
        cov = np.cov((yx - mean.reshape(1, 2)).T)
        if cov.ndim < 2:
            return []
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        projections = (yx - mean) @ principal
        lo = yx[int(np.argmin(projections))]
        hi = yx[int(np.argmax(projections))]
        pts = [(int(lo[0]), int(lo[1])), (int(hi[0]), int(hi[1]))]

    return [(int(r), int(c)) for r, c in pts]


def _average_width(comp_mask: np.ndarray, skeleton: np.ndarray) -> float:
    skel_len = int((comp_mask.astype(bool) & skeleton.astype(bool)).sum())
    area = int(comp_mask.sum())
    if skel_len == 0:
        return max(1.0, math.sqrt(area) / 2.0)
    return max(1.0, area / skel_len)


# ── Stage 3: Intelligent road linking ─────────────────────────────────────────

def _link_road_segments(
    mask: np.ndarray,
    components: List[_RoadComponent],
    obstacle: np.ndarray,
) -> np.ndarray:
    """
    Draw bridge strokes between compatible endpoint pairs from different components.

    A bridge is only drawn when ALL constraints are satisfied:
      ① endpoint distance ≤ ROAD_MAX_CONNECTION_DISTANCE
      ② orientation difference ≤ ROAD_MAX_ANGLE_DIFFERENCE
      ③ straight path does not cross an obstacle pixel
      ④ endpoints broadly face each other (connection direction aligns with orientations)
    """
    max_dist = float(settings.ROAD_MAX_CONNECTION_DISTANCE)
    max_angle = float(settings.ROAD_MAX_ANGLE_DIFFERENCE)

    ep_records: List[Tuple[int, int, _RoadComponent]] = [
        (ep[0], ep[1], comp)
        for comp in components
        for ep in comp.endpoints
    ]

    drawn: set = set()

    for i, (r1, c1, comp1) in enumerate(ep_records):
        for j, (r2, c2, comp2) in enumerate(ep_records):
            if j <= i or comp1.label == comp2.label:
                continue
            if (i, j) in drawn:
                continue

            dist = math.hypot(r2 - r1, c2 - c1)
            if dist > max_dist:
                continue

            if _angle_diff(comp1.orientation, comp2.orientation) > max_angle:
                continue

            if not _endpoints_face_each_other(r1, c1, comp1, r2, c2, comp2):
                continue

            if _path_blocked(r1, c1, r2, c2, obstacle):
                continue

            bridge_w = max(1, int(round((comp1.avg_width + comp2.avg_width) / 2.0)))
            _draw_bridge(mask, r1, c1, r2, c2, bridge_w)
            drawn.add((i, j))

    return mask


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _endpoints_face_each_other(
    r1: int, c1: int, comp1: "_RoadComponent",
    r2: int, c2: int, comp2: "_RoadComponent",
) -> bool:
    """Connection direction must broadly align with both component orientations."""
    dr, dc = r2 - r1, c2 - c1
    if dr == 0 and dc == 0:
        return False
    conn_angle = math.degrees(math.atan2(dc, dr)) % 180.0
    # Relaxed to 45° so curves and slightly misaligned stubs still connect
    threshold = max(settings.ROAD_MAX_ANGLE_DIFFERENCE * 2.0, 45.0)
    return (
        _angle_diff(conn_angle, comp1.orientation) <= threshold
        and _angle_diff(conn_angle, comp2.orientation) <= threshold
    )


def _path_blocked(
    r1: int, c1: int,
    r2: int, c2: int,
    obstacle: np.ndarray,
) -> bool:
    steps = max(abs(r2 - r1), abs(c2 - c1), 1)
    h, w = obstacle.shape
    for t in range(1, steps):
        r = int(round(r1 + (r2 - r1) * t / steps))
        c = int(round(c1 + (c2 - c1) * t / steps))
        if 0 <= r < h and 0 <= c < w and obstacle[r, c]:
            return True
    return False


def _draw_bridge(
    mask: np.ndarray,
    r1: int, c1: int,
    r2: int, c2: int,
    width: int,
) -> None:
    cv2.line(mask, (c1, r1), (c2, r2), 1, thickness=max(1, width))


# ── Stage 4: Road network refinement ──────────────────────────────────────────

def _refine_mask(mask: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """Fill small gaps, remove tiny fragments, smooth boundaries, exclude obstacles."""
    min_area = settings.ROAD_MIN_COMPONENT_AREA

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    clean = np.zeros_like(refined)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lbl] = 1

    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, smooth_k, iterations=1)

    if obstacle.any():
        clean[obstacle > 0] = 0

    return clean.astype(np.uint8)


# ── Stage 5: Chunk-edge continuity ────────────────────────────────────────────

def _handle_chunk_boundary(
    mask: np.ndarray,
    obstacle: np.ndarray,
    chunk,
) -> np.ndarray:
    """
    For every chunk edge that borders another chunk, dilate road pixels
    directionally to reach the tile boundary pixel.

    Strategy: for each applicable edge, apply an anisotropic morphological
    dilation that extends road pixels toward that edge within _EDGE_MARGIN
    columns/rows.  Any road pixel within _EDGE_MARGIN of the boundary will
    touch the boundary after this step.

    When the adjacent chunk does the same on its matching edge, the two road
    polygons will share a boundary coordinate and Shapely's existing
    unary_union dissolves them into one continuous polygon.

    Only non-image-boundary edges are processed (no false extension at the
    border of the overall orthophoto).
    """
    chunk_size = settings.CHUNK_SIZE
    total_cols = (chunk.image_width + chunk_size - 1) // chunk_size
    total_rows = (chunk.image_height + chunk_size - 1) // chunk_size

    core_col_off = chunk.col * chunk_size
    core_row_off = chunk.row * chunk_size
    core_w = min(chunk_size, chunk.image_width - core_col_off)
    core_h = min(chunk_size, chunk.image_height - core_row_off)

    x0 = int(round(core_col_off - chunk.window.col_off))
    y0 = int(round(core_row_off - chunk.window.row_off))
    x1 = min(mask.shape[1], x0 + core_w)
    y1 = min(mask.shape[0], y0 + core_h)

    extended = mask.copy()
    m = _EDGE_MARGIN

    # Top edge — directional dilation upward toward y0
    if chunk.row > 0:
        x_lo, x_hi = x0, x1
        y_lo, y_hi = y0, min(mask.shape[0], y0 + m)
        _dilate_toward_edge(extended, obstacle, x_lo, x_hi, y_lo, y_hi,
                            target_row=y0, direction="up")

    # Bottom edge — directional dilation downward toward y1-1
    if chunk.row < total_rows - 1:
        x_lo, x_hi = x0, x1
        y_lo, y_hi = max(0, y1 - m), y1
        _dilate_toward_edge(extended, obstacle, x_lo, x_hi, y_lo, y_hi,
                            target_row=y1 - 1, direction="down")

    # Left edge — directional dilation leftward toward x0
    if chunk.col > 0:
        x_lo, x_hi = x0, min(mask.shape[1], x0 + m)
        y_lo, y_hi = y0, y1
        _dilate_toward_edge(extended, obstacle, x_lo, x_hi, y_lo, y_hi,
                            target_col=x0, direction="left")

    # Right edge — directional dilation rightward toward x1-1
    if chunk.col < total_cols - 1:
        x_lo, x_hi = max(0, x1 - m), x1
        y_lo, y_hi = y0, y1
        _dilate_toward_edge(extended, obstacle, x_lo, x_hi, y_lo, y_hi,
                            target_col=x1 - 1, direction="right")

    return extended


def _dilate_toward_edge(
    mask: np.ndarray,
    obstacle: np.ndarray,
    x_lo: int, x_hi: int,
    y_lo: int, y_hi: int,
    target_row: int = -1,
    target_col: int = -1,
    direction: str = "right",
) -> None:
    """
    For each line (row or column) in the strip [y_lo:y_hi, x_lo:x_hi]:
    if any road pixel exists, draw a line from the nearest road pixel to the
    target edge coordinate, provided the path is not obstacle-blocked.

    This ensures roads that stop short of the tile boundary are extended to it
    so Shapely can join them with the matching road from the adjacent chunk.
    """
    h, w = mask.shape

    if direction in ("left", "right"):
        # Extend horizontally: iterate over each row in the strip
        for abs_r in range(max(0, y_lo), min(h, y_hi)):
            row_strip = mask[abs_r, max(0, x_lo): min(w, x_hi)]
            road_cols = np.nonzero(row_strip)[0]
            if len(road_cols) == 0:
                continue
            if direction == "right":
                src_c = max(0, x_lo) + int(road_cols[-1])   # rightmost road pixel
                dst_c = min(w - 1, target_col)
            else:
                src_c = max(0, x_lo) + int(road_cols[0])    # leftmost road pixel
                dst_c = max(0, target_col)

            if dst_c == src_c:
                continue
            if _path_blocked(abs_r, src_c, abs_r, dst_c, obstacle):
                continue
            cv2.line(mask, (src_c, abs_r), (dst_c, abs_r), 1, thickness=3)

    else:
        # Extend vertically: iterate over each column in the strip
        for abs_c in range(max(0, x_lo), min(w, x_hi)):
            col_strip = mask[max(0, y_lo): min(h, y_hi), abs_c]
            road_rows = np.nonzero(col_strip)[0]
            if len(road_rows) == 0:
                continue
            if direction == "down":
                src_r = max(0, y_lo) + int(road_rows[-1])   # bottommost road pixel
                dst_r = min(h - 1, target_row)
            else:
                src_r = max(0, y_lo) + int(road_rows[0])    # topmost road pixel
                dst_r = max(0, target_row)

            if dst_r == src_r:
                continue
            if _path_blocked(src_r, abs_c, dst_r, abs_c, obstacle):
                continue
            cv2.line(mask, (abs_c, src_r), (abs_c, dst_r), 1, thickness=3)


# ── Obstacle mask ──────────────────────────────────────────────────────────────

def _build_obstacle_mask(
    road_mask: np.ndarray,
    building_mask: Optional[np.ndarray],
    water_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Union of building and water masks — road bridges must not cross these areas."""
    h, w = road_mask.shape
    obstacle = np.zeros((h, w), dtype=np.uint8)

    if building_mask is not None and building_mask.shape == road_mask.shape:
        obstacle = cv2.bitwise_or(obstacle, (building_mask > 0).astype(np.uint8))

    if water_mask is not None and water_mask.shape == road_mask.shape:
        obstacle = cv2.bitwise_or(obstacle, (water_mask > 0).astype(np.uint8))

    return obstacle
