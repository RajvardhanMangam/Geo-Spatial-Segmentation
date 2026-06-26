"""
building_separator.py

Adaptive Building Instance Separation (ABIS)

Full pipeline
-------------
Part 1  – Morphological cleaning, connected components, feature extraction
Part 2  – Component graph construction and cluster detection
Part 3  – ROI cropping, edge map, distance transform, marker generation
Part 4  – Marker-controlled watershed split + instance validation
Part 5  – Global mask projection, split_clusters orchestrator
Part 6  – Region Adjacency Graph (RAG)
Part 7  – Merge incorrect watershed splits
Part 8  – Assign persistent building IDs
Part 9  – Roof feature extraction
Part 10 – Public API  (extract_instances / separate_buildings)

Usage
-----
    separator = BuildingSeparator()
    inventory = separator.extract_instances(rgb_image, building_mask)

    for building in inventory:
        print(building["building_id"], building["bbox"])
        # keys: building_id, bbox, mask, polygon, roof_features
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple
import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# DATA CLASS
# ===========================================================================

@dataclass
class BuildingComponent:
    """
    Represents one connected component extracted
    from the semantic building mask.
    """

    component_id: int
    """Label index from cv2.connectedComponentsWithStats."""

    area: int
    """Pixel area of the component."""

    bbox: Tuple[int, int, int, int]
    """(x, y, w, h) bounding box in full-image coordinates."""

    centroid: Tuple[float, float]
    """(cx, cy) centroid in full-image coordinates."""

    mask: np.ndarray
    """Full-image binary mask (uint8) for this component."""

    contour: np.ndarray
    """OpenCV contour in full-image coordinates."""

    rgb_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Mean RGB colour of pixels inside the component."""

    edge_density: float = 0.0
    """Fraction of component pixels that fall on an edge."""

    texture_score: float = 0.0
    """Laplacian variance used as a texture proxy."""

    orientation: float = 0.0
    """Major-axis angle (degrees) from fitted ellipse."""

    solidity: float = 0.0
    """contour_area / convex_hull_area."""

    cluster_id: int = -1
    """Cluster this component was assigned to."""


# ===========================================================================
# MAIN CLASS
# ===========================================================================

class BuildingSeparator:
    """
    End-to-end building instance separator.

    Takes a binary semantic building mask (from a SegFormer or similar model)
    and an aligned RGB image, and returns a list of individual building
    instances, each with its own binary mask, bounding box, polygon, and
    basic roof features.

    Parameters
    ----------
    min_area : int
        Minimum pixel area for a connected component to be kept (default 75).
    kernel_size : int
        Side length of the morphological structuring element (default 3).
    """

    def __init__(
        self,
        min_area: int = 75,
        kernel_size: int = 3,
        max_distance: float = 10.0,
        max_colour: float = 45.0,
        max_orientation: float = 25.0,
        side_overlap_ratio: float = 0.25,
        merge_max_colour: float = 25.0,
        merge_min_shared_boundary: int = 15,
        roof_split_enabled: bool = True,
        roof_split_clusters: int = 4,
        roof_split_min_area: int = 120,
        roof_split_min_color_distance: float = 35.0,
    ):
        self.min_area = min_area
        self.max_distance = max_distance
        self.max_colour = max_colour
        self.max_orientation = max_orientation
        self.side_overlap_ratio = side_overlap_ratio
        self.merge_max_colour = merge_max_colour
        self.merge_min_shared_boundary = merge_min_shared_boundary
        self.roof_split_enabled = roof_split_enabled
        self.roof_split_clusters = roof_split_clusters
        self.roof_split_min_area = roof_split_min_area
        self.roof_split_min_color_distance = roof_split_min_color_distance
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

    # =======================================================================
    # PART 1 – MORPHOLOGICAL CLEANING & CONNECTED COMPONENTS
    # =======================================================================

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove small speckles (opening) then close small holes (closing).

        Parameters
        ----------
        mask : np.ndarray
            Binary mask, dtype uint8 (0/1 or 0/255).

        Returns
        -------
        np.ndarray
            Cleaned binary mask, same shape and dtype.
        """
        original = (mask > 0).astype(np.uint8)
        mask = cv2.morphologyEx(original, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return np.logical_and(mask > 0, original > 0).astype(np.uint8)

    def extract_components(
        self,
        rgb: np.ndarray,
        building_mask: np.ndarray,
    ) -> List[BuildingComponent]:
        """
        Clean the mask and extract per-component features.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image aligned with building_mask, shape (H, W, 3), uint8.
        building_mask : np.ndarray
            Binary building mask, shape (H, W).

        Returns
        -------
        List[BuildingComponent]
            One entry per component that passes the min_area threshold.
        """
        source_mask = (building_mask > 0).astype(np.uint8)
        mask = self.clean_mask(source_mask)
        flood = mask.copy()

        h, w = flood.shape

        mask_ff = np.zeros(
            (h+2,w+2),
            np.uint8
        )

        cv2.floodFill(
            flood,
            mask_ff,
            (0,0),
            255
        )

        holes = cv2.bitwise_not(flood)

        mask = np.logical_and((mask | holes) > 0, source_mask > 0).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        components: List[BuildingComponent] = []

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])

            component_mask = (labels == label).astype(np.uint8)
            roi_mask = component_mask[y:y + h, x:x + w]
            roi_rgb  = rgb[y:y + h, x:x + w]

            contours, _ = cv2.findContours(
                roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            contour = contour + np.array([[x, y]])  # shift to full-image coords

            rgb_pixels = roi_rgb[roi_mask == 1]
            rgb_mean = rgb_pixels.mean(axis=0) if len(rgb_pixels) else np.zeros(3)

            orientation = 0.0
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                orientation = ellipse[2]

            hull = cv2.convexHull(contour)
            hull_area    = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            solidity = contour_area / hull_area if hull_area > 0 else 1.0

            components.append(
                BuildingComponent(
                    component_id=label,
                    area=area,
                    bbox=(x, y, w, h),
                    centroid=(
                        float(centroids[label][0]),
                        float(centroids[label][1]),
                    ),
                    mask=component_mask,
                    contour=contour,
                    rgb_mean=rgb_mean,
                    orientation=float(orientation),
                    solidity=float(solidity),
                )
            )

        logger.debug("[Separator] Components found: %d", len(components))
        print(f"[Separator] Components Found : {len(components)}")
        return components

    def visualize_components(
        self,
        rgb: np.ndarray,
        components: List[BuildingComponent],
    ) -> np.ndarray:
        """Draw bounding boxes and component IDs on a copy of the image."""
        image = rgb.copy()
        rng = np.random.default_rng(42)

        for comp in components:
            color = tuple(int(c) for c in rng.integers(0, 255, size=3))
            x, y, w, h = comp.bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image, str(comp.component_id),
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA,
            )
        return image

    # =======================================================================
    # PART 2 – COMPONENT GRAPH & CLUSTERING
    # =======================================================================

    # def component_distance(
    #     self,
    #     c1: BuildingComponent,
    #     c2: BuildingComponent,
    # ) -> float:
    #     """Euclidean distance between two component centroids."""
    #     x1, y1 = c1.centroid
    #     x2, y2 = c2.centroid
    #     return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def color_distance(
        self,
        c1: BuildingComponent,
        c2: BuildingComponent,
    ) -> float:
        """L2 distance between mean RGB colours."""
        return float(np.linalg.norm(c1.rgb_mean - c2.rgb_mean))

    def orientation_difference(
        self,
        c1: BuildingComponent,
        c2: BuildingComponent,
    ) -> float:
        """Smallest angular difference between two orientations (0–90°)."""
        diff = abs(c1.orientation - c2.orientation)
        return min(diff, 180.0 - diff)

    def bbox_distance(
        self,
        c1: BuildingComponent,
        c2: BuildingComponent,
    ) -> float:
        """Pixel gap between the two bounding boxes (0 if overlapping)."""
        x1, y1, w1, h1 = c1.bbox
        x2, y2, w2, h2 = c2.bbox
        dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        return float(np.sqrt(dx * dx + dy * dy))

    def side_by_side_overlap(
        self,
        c1: BuildingComponent,
        c2: BuildingComponent,
    ) -> bool:
        """Return true only when bboxes are adjacent along a real shared side."""
        x1, y1, w1, h1 = c1.bbox
        x2, y2, w2, h2 = c2.bbox

        x_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        y_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        horizontal_touch = (
            x_gap <= self.max_distance
            and y_overlap / max(1, min(h1, h2)) >= self.side_overlap_ratio
        )
        vertical_touch = (
            y_gap <= self.max_distance
            and x_overlap / max(1, min(w1, w2)) >= self.side_overlap_ratio
        )
        return horizontal_touch or vertical_touch

    def should_connect(
        self,
        c1: BuildingComponent,
        c2: BuildingComponent,
    ) -> bool:
        """
        Decide whether two components belong to the same cluster.

        Returns False early if any criterion is violated.
        """
        if not self.side_by_side_overlap(c1, c2):
            return False
        if self.color_distance(c1, c2) > self.max_colour:
            return False
        if self.orientation_difference(c1, c2) > self.max_orientation:
            return False
        if self.bbox_distance(c1, c2) > self.max_distance:
            return False
        return True

    def build_component_graph(
        self,
        components: List[BuildingComponent],
    ) -> dict:
        """
        Build an undirected adjacency graph over components.

        Returns
        -------
        dict
            Mapping component_id → list of neighbouring component_ids.
        """
        graph: dict = {c.component_id: [] for c in components}

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                c1, c2 = components[i], components[j]
                if self.should_connect(c1, c2):
                    graph[c1.component_id].append(c2.component_id)
                    graph[c2.component_id].append(c1.component_id)

        logger.debug("[Separator] Graph nodes: %d", len(graph))
        print(f"[Separator] Graph Nodes : {len(graph)}")
        return graph

    # Kept as an alias so existing call-sites using build_graph still work.
    def build_graph(self, components: List[BuildingComponent]) -> dict:
        return self.build_component_graph(components)

    def cluster_components(
        self,
        components: List[BuildingComponent],
    ) -> List[List[BuildingComponent]]:
        """
        Partition components into clusters via BFS on the component graph.

        Returns
        -------
        List[List[BuildingComponent]]
            Each inner list is one cluster.
        """
        graph    = self.build_component_graph(components)
        comp_map = {c.component_id: c for c in components}

        visited    = set()
        clusters   = []
        cluster_id = 0

        for component in components:
            if component.component_id in visited:
                continue

            queue           = deque([component.component_id])
            current_cluster = []

            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)

                comp = comp_map[node]
                comp.cluster_id = cluster_id
                current_cluster.append(comp)

                for neighbour in graph[node]:
                    if neighbour not in visited:
                        queue.append(neighbour)

            clusters.append(current_cluster)
            cluster_id += 1

        logger.debug("[Separator] Clusters: %d", len(clusters))
        print(f"[Separator] Clusters : {len(clusters)}")
        return clusters

    def visualize_clusters(
        self,
        rgb: np.ndarray,
        clusters: List[List[BuildingComponent]],
    ) -> np.ndarray:
        """Draw each cluster's components in a shared random colour."""
        image = rgb.copy()
        rng   = np.random.default_rng(1)

        for cluster in clusters:
            color = tuple(int(x) for x in rng.integers(0, 255, size=3))
            for comp in cluster:
                x, y, w, h = comp.bbox
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    image, f"C{comp.cluster_id}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1,
                )
        return image

    # =======================================================================
    # PART 3 – ROI CROPPING, EDGE MAP, DISTANCE MAP, MARKER GENERATION
    # =======================================================================

    def crop_cluster(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        cluster: List[BuildingComponent],
        pad: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Return the RGB and mask crops that cover all components in a cluster.

        Returns
        -------
        (rgb_roi, mask_roi, (x_offset, y_offset))
        """
        xs, ys = [], []
        for comp in cluster:
            x, y, w, h = comp.bbox
            xs.extend([x, x + w])
            ys.extend([y, y + h])

        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(rgb.shape[1], max(xs) + pad)
        y2 = min(rgb.shape[0], max(ys) + pad)

        return rgb[y1:y2, x1:x2], mask[y1:y2, x1:x2], (x1, y1)

    def compute_edge_map(self, rgb: np.ndarray) -> np.ndarray:
        """
        Combine Canny edges and Sobel gradient magnitude into one edge map.

        Returns
        -------
        np.ndarray
            uint8 edge map, same H×W as input.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        canny = cv2.Canny(gray, 40, 120)

        grad_x   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
        grad_y   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
        gradient = gradient.astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3,3)
        )

        morph = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            kernel
        )

        edge = np.maximum.reduce([
            canny,
            gradient,
            morph
        ])

        return edge

    def compute_gradient_map(self, rgb: np.ndarray) -> np.ndarray:
        """
        Return the morphological gradient (dilation − erosion) in grayscale.

        Useful as an additional boundary signal alongside the Sobel gradient.
        """
        gray   = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    def compute_distance_map(self, mask: np.ndarray) -> np.ndarray:
        """
        Euclidean distance transform of the binary mask.

        Returns
        -------
        np.ndarray
            float32 distance map; peaks correspond to likely building centres.
        """
        return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

    def detect_local_peaks(
        self,
        distance: np.ndarray,
        min_distance: int = 7,
    ) -> np.ndarray:
        """
        Detect local maxima in the distance map.

        Parameters
        ----------
        distance : np.ndarray
            Float32 distance transform.
        min_distance : int
            Minimum separation between peaks in pixels.

        Returns
        -------
        np.ndarray
            uint8 binary image with 1 at each peak location.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * min_distance + 1, 2 * min_distance + 1),
        )
        dilated = cv2.dilate(distance, kernel)
        peaks   = (distance == dilated) & (distance > 0)
        return peaks.astype(np.uint8)

    def generate_markers(
        self,
        distance: np.ndarray,
        edge: np.ndarray,
    ) -> np.ndarray:
        """
        Suppress the distance map at strong edges, then threshold and label.

        Returns
        -------
        np.ndarray
            int32 marker array for cv2.watershed.
        """
        dist = distance.copy()
        dist[edge > 80] *= 0.3

        peaks = self.detect_local_peaks(
            dist
        )

        _, markers = cv2.connectedComponents(
            peaks
        )
        return markers.astype(np.int32)

    # =======================================================================
    # PART 4 – WATERSHED SPLIT & INSTANCE VALIDATION
    # =======================================================================

    def watershed_split(
        self,
        rgb_roi: np.ndarray,
        mask_roi: np.ndarray,
        markers: np.ndarray,
    ) -> np.ndarray:
        """
        Run marker-controlled watershed on the cluster ROI.

        Returns
        -------
        np.ndarray
            int32 label map; background=1, boundaries=-1, regions≥2.
        """
        image   = rgb_roi.copy()
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        markers = markers.copy().astype(np.int32)
        markers += 1                        # reserve 1 for background
        markers[mask_roi == 0] = 0          # unknown pixels → 0

        return cv2.watershed(image, markers)

    def validate_instances(
        self,
        markers: np.ndarray,
        source_mask: np.ndarray,
        min_area: int = 80,
    ) -> List[np.ndarray]:
        """
        Filter watershed regions by area and solidity.

        Parameters
        ----------
        markers : np.ndarray
            int32 watershed output.
        min_area : int
            Minimum pixel count to keep a region.

        Returns
        -------
        List[np.ndarray]
            List of uint8 binary masks, one per accepted instance.
        """
        instances = []

        for label in np.unique(markers):
            if label <= 1:      # watershed boundary, unknown, and background
                continue

            mask = np.logical_and(markers == label, source_mask > 0).astype(np.uint8)
            if np.sum(mask) < min_area:
                continue

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour      = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(contour)
            hull_area    = cv2.contourArea(cv2.convexHull(contour))

            if hull_area == 0:
                continue
            solidity = contour_area / hull_area

            print(
                "Area:",
                np.sum(mask),
                "Solidity:",
                solidity
            )
            if contour_area / hull_area < 0.30:
                continue
            x, y, w, h = cv2.boundingRect(contour)

            aspect = max(w, h) / max(min(w, h), 1)

            if aspect > 12:
                continue

            instances.append(mask)

        return instances

    # =======================================================================
    # PART 5 – CLUSTER-LEVEL ORCHESTRATION
    # =======================================================================

    def split_cluster(
        self,
        rgb: np.ndarray,
        building_mask: np.ndarray,
        cluster: List[BuildingComponent],
    ) -> List[np.ndarray]:
        """
        Run the full watershed pipeline on one cluster and return global masks.

        Returns
        -------
        List[np.ndarray]
            Binary masks in full-image coordinates, one per watershed instance.
        """
        if len(cluster) == 1:
            return [cluster[0].mask.astype(np.uint8)]

        rgb_roi, mask_roi, (ox, oy) = self.crop_cluster(rgb, building_mask, cluster)

        edge     = self.compute_edge_map(rgb_roi)
        distance = self.compute_distance_map(mask_roi)
        markers  = self.generate_markers(distance, edge)
        watershed = self.watershed_split(rgb_roi, mask_roi, markers)
        instances = self.validate_instances(watershed, mask_roi)
        cleaned = []

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3,3)
        )

        for mask in instances:

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            cleaned.append(mask)

        instances = cleaned

        global_masks = []
        for instance in instances:
            global_mask = np.zeros_like(building_mask, dtype=np.uint8)
            h, w        = instance.shape
            global_mask[oy:oy + h, ox:ox + w] = instance
            global_masks.append(global_mask)

        if not global_masks:
            return [component.mask.astype(np.uint8) for component in cluster]

        return global_masks

    def split_clusters(
        self,
        rgb: np.ndarray,
        building_mask: np.ndarray,
        clusters: List[List[BuildingComponent]],
    ) -> List[np.ndarray]:
        """
        Apply split_cluster to every cluster and collect all instance masks.
        """
        all_instances = []
        for cluster in clusters:
            cluster_instances = self.split_cluster(rgb, building_mask, cluster)
            for instance in cluster_instances:
                all_instances.extend(self.split_by_roof_material(rgb, instance))

        if not all_instances:
            all_instances = self._component_masks_from_clusters(clusters)

        logger.debug("[Separator] Final instances after split: %d", len(all_instances))
        print(f"[Separator] Final Instances : {len(all_instances)}")
        return all_instances

    def split_by_roof_material(
        self,
        rgb: np.ndarray,
        building_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """Split one building blob into large roof-material regions."""
        building_mask = (building_mask > 0).astype(np.uint8)
        if not self.roof_split_enabled:
            return [building_mask]

        area = int(np.sum(building_mask))
        if area < max(self.roof_split_min_area * 2, self.min_area * 2):
            return [building_mask]

        x, y, w, h = cv2.boundingRect(building_mask)
        rgb_roi = rgb[y:y + h, x:x + w]
        mask_roi = building_mask[y:y + h, x:x + w]

        smoothed = cv2.pyrMeanShiftFiltering(rgb_roi, sp=15, sr=38)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)

        yy, xx = np.indices(mask_roi.shape)
        pixels_lab = lab[mask_roi > 0].astype(np.float32)
        coords = np.column_stack([xx[mask_roi > 0], yy[mask_roi > 0]]).astype(np.float32)
        spatial_scale = 0.05
        pixels = np.column_stack([pixels_lab, coords * spatial_scale]).astype(np.float32)
        if len(pixels) < self.roof_split_min_area * 2:
            return [building_mask]

        k = min(self.roof_split_clusters, max(2, len(pixels) // self.roof_split_min_area))
        if k < 2:
            return [building_mask]

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                1.0,
            ),
            3,
            cv2.KMEANS_PP_CENTERS,
        )

        centers_lab = centers[:, :3].astype(np.float32)
        if self._max_center_distance(centers_lab) < self.roof_split_min_color_distance:
            return [building_mask]

        label_roi = np.full(mask_roi.shape, 255, dtype=np.uint8)
        label_roi[mask_roi > 0] = labels.reshape(-1).astype(np.uint8)
        label_roi = cv2.medianBlur(label_roi, 11)
        label_roi[mask_roi == 0] = 255

        instances: List[np.ndarray] = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        for label in range(k):
            material_mask = (label_roi == label).astype(np.uint8)
            material_mask = cv2.morphologyEx(material_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            material_mask = cv2.morphologyEx(material_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            material_mask = np.logical_and(material_mask > 0, mask_roi > 0).astype(np.uint8)

            num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
                material_mask,
                connectivity=8,
            )
            for cc in range(1, num_labels):
                cc_area = int(stats[cc, cv2.CC_STAT_AREA])
                if cc_area < self.roof_split_min_area:
                    continue
                local_instance = (cc_labels == cc).astype(np.uint8)
                global_instance = np.zeros_like(building_mask, dtype=np.uint8)
                global_instance[y:y + h, x:x + w] = local_instance
                instances.append(global_instance)

        covered = np.zeros_like(building_mask, dtype=np.uint8)
        for instance in instances:
            covered = np.logical_or(covered, instance).astype(np.uint8)

        if len(instances) < 2:
            return [building_mask]

        instances = self._assign_remainder_to_instances(building_mask, instances)
        if not self._split_has_clear_boundary(rgb, building_mask, instances):
            return [building_mask]

        return instances

    def _assign_remainder_to_instances(
        self,
        source_mask: np.ndarray,
        instances: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Attach small unassigned pixels to the nearest large roof region."""
        label_map = np.zeros(source_mask.shape, dtype=np.int32)
        for idx, instance in enumerate(instances, start=1):
            label_map[instance > 0] = idx

        remainder = np.logical_and(source_mask > 0, label_map == 0)
        if not np.any(remainder):
            return instances

        nearest_label = np.zeros(source_mask.shape, dtype=np.int32)
        nearest_dist = np.full(source_mask.shape, np.inf, dtype=np.float32)

        for idx, instance in enumerate(instances, start=1):
            inv = (instance == 0).astype(np.uint8)
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
            update = dist < nearest_dist
            nearest_dist[update] = dist[update]
            nearest_label[update] = idx

        label_map[remainder] = nearest_label[remainder]

        merged = []
        for idx in range(1, len(instances) + 1):
            merged.append((label_map == idx).astype(np.uint8))
        return merged

    def _split_has_clear_boundary(
        self,
        rgb: np.ndarray,
        source_mask: np.ndarray,
        instances: List[np.ndarray],
    ) -> bool:
        """Accept a split only when adjacent regions differ clearly in colour."""
        means = []
        for instance in instances:
            pixels = rgb[instance > 0]
            if len(pixels) == 0:
                return False
            means.append(pixels.mean(axis=0).astype(np.float32))

        max_colour_distance = self._max_center_distance(np.asarray(means, dtype=np.float32))
        if max_colour_distance < self.roof_split_min_color_distance:
            return False

        largest = max(int(np.sum(instance)) for instance in instances)
        if largest / max(int(np.sum(source_mask)), 1) > 0.92:
            return False

        return True

    def _large_components(self, mask: np.ndarray) -> List[np.ndarray]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        components = []
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= self.roof_split_min_area:
                components.append((labels == label).astype(np.uint8))
        return components

    def _max_center_distance(self, centers: np.ndarray) -> float:
        max_distance = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                max_distance = max(
                    max_distance,
                    float(np.linalg.norm(centers[i] - centers[j])),
                )
        return max_distance

    def _component_masks_from_clusters(
        self,
        clusters: List[List[BuildingComponent]],
    ) -> List[np.ndarray]:
        """Return component masks when watershed cannot produce valid instances."""
        return [
            component.mask.astype(np.uint8)
            for cluster in clusters
            for component in cluster
        ]

    # =======================================================================
    # PART 6 – REGION ADJACENCY GRAPH (RAG)
    # =======================================================================

    def build_rag(
        self,
        instances: List[np.ndarray],
    ) -> defaultdict:
        """
        Build adjacency graph between separated instances.

        Two instances are adjacent if they touch after a 1-pixel dilation.

        Returns
        -------
        defaultdict(list)
            Mapping instance index → list of adjacent instance indices.
        """
        graph  = defaultdict(list)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for i, inst_i in enumerate(instances):
            dilated_i = cv2.dilate(inst_i.astype(np.uint8), kernel, iterations=2)

            for j in range(i + 1, len(instances)):
                if np.any(np.logical_and(dilated_i, instances[j])):
                    graph[i].append(j)
                    graph[j].append(i)

        return graph

    # =======================================================================
    # PART 7 – MERGE INCORRECT SPLITS
    # =======================================================================

    def compute_merge_score(
        self,
        m1: np.ndarray,
        m2: np.ndarray,
        rgb: np.ndarray,
    ) -> float:
        """
        Compute a similarity score between two adjacent instances.

        Score = 0.5 × colour_similarity + 0.5 × boundary_density
        (higher → more likely to merge).

        Returns
        -------
        float
            Score in [0, 1]; threshold at 0.5 to decide merge.
        """
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        d1       = cv2.dilate(m1.astype(np.uint8), kernel)
        boundary = np.logical_and(d1, m2)
        boundary_density = float(np.sum(boundary)) / max(
            float(np.sum(m1) + np.sum(m2)), 1.0
        )

        rgb1 = rgb[m1 > 0]
        rgb2 = rgb[m2 > 0]
        if len(rgb1) == 0 or len(rgb2) == 0:
            return 0.0

        colour_dist       = float(np.linalg.norm(rgb1.mean(0) - rgb2.mean(0)))
        colour_similarity = max(0.0, 1.0 - colour_dist / 255.0)

        return (
                0.6 * colour_similarity +
                0.4 * boundary_density
            )

    def should_merge_instances(
        self,
        m1: np.ndarray,
        m2: np.ndarray,
        rgb: np.ndarray,
        max_colour: float = 25.0,
        min_shared_boundary: int = 15,
    ) -> bool:
        """
        Decide whether two adjacent watershed instances should be re-merged.

        Checks shared boundary length and colour distance.
        """
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        d1       = cv2.dilate(m1.astype(np.uint8), kernel)
        boundary = np.logical_and(d1, m2)

        if np.sum(boundary) < min_shared_boundary:
            return False

        rgb1 = rgb[m1 > 0]
        rgb2 = rgb[m2 > 0]
        if len(rgb1) == 0 or len(rgb2) == 0:
            return False

        if np.linalg.norm(rgb1.mean(0) - rgb2.mean(0)) > max_colour:
            return False

        return True

    def merge_instances(
        self,
        rgb: np.ndarray,
        instances: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Greedily merge over-segmented watershed regions using the RAG.

        Returns
        -------
        List[np.ndarray]
            Merged binary masks.
        """
        graph   = self.build_rag(instances)
        visited = set()
        merged  = []

        for i in range(len(instances)):
            if i in visited:
                continue

            current = instances[i].copy()
            queue   = deque([i])

            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)

                for neigh in graph[node]:
                    if neigh in visited:
                        continue
                    if self.should_merge_instances(
                        current,
                        instances[neigh],
                        rgb,
                        max_colour=self.merge_max_colour,
                        min_shared_boundary=self.merge_min_shared_boundary,
                    ):
                        current = np.logical_or(current, instances[neigh]).astype(np.uint8)
                        queue.append(neigh)

            merged.append(current)

        logger.debug("[Separator] After merge: %d", len(merged))
        print(f"[Separator] After Merge : {len(merged)}")
        return merged

    # =======================================================================
    # PART 8 – ASSIGN PERSISTENT BUILDING IDs
    # =======================================================================

    def assign_building_ids(
        self,
        instances: List[np.ndarray],
    ) -> List[dict]:
        """
        Assign a sequential 1-based ID to each instance.

        Returns
        -------
        List[dict]
            Keys: building_id, bbox (x,y,w,h), mask, polygon.
        """
        inventory = []

        for idx, mask in enumerate(instances, start=1):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue

            contour     = max(contours, key=cv2.contourArea)
            x, y, w, h  = cv2.boundingRect(contour)
            inventory.append({
                "building_id": f"B{idx:06d}",
                "bbox":        (x, y, w, h),
                "mask":        mask,
                "polygon":     contour.reshape(-1, 2).tolist(),
            })

        return inventory

    # =======================================================================
    # PART 9 – ROOF FEATURE EXTRACTION
    # =======================================================================

    def extract_roof_features(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Extract a simple feature vector from a building's roof pixels.

        Returns
        -------
        np.ndarray
            [mean_R, mean_G, mean_B, texture_variance, roof_angle]
        """
        pixels   = rgb[mask > 0]
        mean_rgb = pixels.mean(axis=0) if len(pixels) else np.zeros(3)

        gray    = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        texture = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        angle = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:
                rect  = cv2.minAreaRect(contour)
                angle = float(rect[2])

        area = np.sum(mask)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        return np.array([
        mean_rgb[0],
        mean_rgb[1],
        mean_rgb[2],
        texture,
        angle,
        area,
        perimeter
        ], dtype=np.float32)
    # =======================================================================
    # PART 10 – PUBLIC API
    # =======================================================================

    def extract_instances(
        self,
        rgb: np.ndarray,
        semantic_mask: np.ndarray,
    ) -> List[dict]:
        """
        Full pipeline: semantic mask → individual building instances.

        This is the **primary public entry point**.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image, shape (H, W, 3), dtype uint8.
        semantic_mask : np.ndarray
            Binary building mask, shape (H, W).
            Non-zero pixels are treated as building.

        Returns
        -------
        List[dict]
            One dict per building with keys:
              - ``building_id``    – int, 1-based sequential index
              - ``bbox``          – (x, y, w, h) in pixel coordinates
              - ``mask``          – uint8 binary mask, shape (H, W)
              - ``polygon``       – OpenCV contour array
              - ``roof_features`` – float32 array [R, G, B, texture, angle]

        Example
        -------
        >>> sep = BuildingSeparator()
        >>> buildings = sep.extract_instances(rgb_image, building_mask)
        >>> for b in buildings:
        ...     print(b["building_id"], b["bbox"])
        """
        components = self.extract_components(rgb, semantic_mask)
        clusters   = self.cluster_components(components)
        instances  = self.split_clusters(rgb, semantic_mask, clusters)
        instances  = self.merge_instances(rgb, instances)
        inventory  = self.assign_building_ids(instances)
        for building in inventory:
            building["confidence"] = 1.0

        # Attach roof features to each entry
        for entry in inventory:
            entry["roof_features"] = self.extract_roof_features(
                rgb, entry["mask"]
            )

        logger.info("[Separator] extract_instances complete: %d buildings", len(inventory))
        return inventory

    # Alias so the original separate_buildings call-sites still work.
    def separate_buildings(
        self,
        rgb: np.ndarray,
        semantic_mask: np.ndarray,
    ) -> List[dict]:
        """Alias for :meth:`extract_instances` (backwards-compatible)."""
        return self.extract_instances(rgb, semantic_mask)
