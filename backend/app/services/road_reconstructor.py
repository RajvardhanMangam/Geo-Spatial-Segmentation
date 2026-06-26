"""Adaptive Semantic Road Graph Reconstruction (ASRG)."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RoadEndpoint:
    point: tuple[int, int]
    direction: tuple[float, float]
    radius: float


class RoadReconstructor:
    """Reconnect short road gaps using skeleton endpoints and semantic costs."""

    def __init__(
        self,
        max_distance: int = 40,
        max_angle: float = 20.0,
        min_probability: float = 0.50,
        min_template_score: float = 0.80,
        max_width_difference: float = 0.30,
        max_path_cost: float = 6.0,
    ):
        self.max_distance = int(max_distance)
        self.max_angle = float(max_angle)
        self.min_probability = float(min_probability)
        self.min_template_score = float(min_template_score)
        self.max_width_difference = float(max_width_difference)
        self.max_path_cost = float(max_path_cost)

    def reconstruct(
        self,
        road_mask: np.ndarray,
        road_probability: np.ndarray,
        semantic_mask: np.ndarray,
        building_class_ids: set[int],
        water_class_ids: set[int],
        rgb: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a road mask with validated endpoint gaps filled."""
        road_mask = (road_mask > 0).astype(np.uint8)
        if int(road_mask.sum()) == 0:
            return road_mask

        cleaned = self._clean_road_mask(road_mask)
        skeleton = self._skeletonize(cleaned)
        endpoints = self._find_endpoints(skeleton, cleaned)
        if len(endpoints) < 2:
            return cleaned

        blockers = self._blocker_mask(
            road_mask,
            road_probability,
            semantic_mask,
            building_class_ids,
            water_class_ids,
        )
        cost_map = self._semantic_cost_map(
            cleaned,
            road_probability,
            semantic_mask,
            building_class_ids,
            water_class_ids,
            rgb,
        )

        reconstructed = cleaned.copy()
        used_pairs: set[tuple[int, int]] = set()

        for i, start in enumerate(endpoints):
            candidates = []
            for j, end in enumerate(endpoints):
                if i >= j or (i, j) in used_pairs:
                    continue
                candidate = self._score_candidate(
                    start, end, road_probability, blockers
                )
                if candidate is not None:
                    candidates.append((candidate, j, end))

            candidates.sort(reverse=True, key=lambda item: item[0])
            for _, j, end in candidates[:3]:
                path = self._least_cost_path(start.point, end.point, cost_map)
                if not path or self._path_crosses_blockers(path, blockers):
                    continue
                if not self._validate_path(
                    path, start, end, road_probability, cost_map
                ):
                    continue

                radius = self._connection_radius(start, end)
                self._draw_path(reconstructed, path, radius)
                used_pairs.add((i, j))
                break

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _clean_road_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        filtered = np.zeros_like(cleaned)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= 20:
                filtered[labels == label] = 1
        return filtered

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Morphological skeletonization implemented with OpenCV primitives."""
        img = (mask > 0).astype(np.uint8)
        skeleton = np.zeros_like(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while cv2.countNonZero(img) > 0:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(img, opened))
            img = eroded

        return skeleton

    def _find_endpoints(
        self,
        skeleton: np.ndarray,
        road_mask: np.ndarray,
    ) -> list[RoadEndpoint]:
        distance = cv2.distanceTransform(road_mask.astype(np.uint8), cv2.DIST_L2, 5)
        endpoints: list[RoadEndpoint] = []

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            road_mask.astype(np.uint8), connectivity=8
        )
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < 20:
                continue
            ys, xs = np.where(labels == label)
            component_endpoints = self._component_axis_endpoints(xs, ys, distance)
            endpoints.extend(component_endpoints)
        return endpoints

    def _component_axis_endpoints(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        distance: np.ndarray,
    ) -> list[RoadEndpoint]:
        if len(xs) < 2:
            return []

        points = np.column_stack([xs, ys]).astype(np.float32)
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0].astype(np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return []
        axis /= norm

        projections = centered @ axis
        first = points[projections <= projections.min() + 1.0].mean(axis=0)
        second = points[projections >= projections.max() - 1.0].mean(axis=0)
        first_direction = -axis
        second_direction = axis

        return [
            RoadEndpoint(
                (int(round(first[0])), int(round(first[1]))),
                (float(first_direction[0]), float(first_direction[1])),
                self._local_radius(distance, first),
            ),
            RoadEndpoint(
                (int(round(second[0])), int(round(second[1]))),
                (float(second_direction[0]), float(second_direction[1])),
                self._local_radius(distance, second),
            ),
        ]

    def _local_radius(self, distance: np.ndarray, point: np.ndarray, radius: int = 4) -> float:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        y0 = max(0, y - radius)
        y1 = min(distance.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(distance.shape[1], x + radius + 1)
        return float(distance[y0:y1, x0:x1].max())

    def _score_candidate(
        self,
        start: RoadEndpoint,
        end: RoadEndpoint,
        road_probability: np.ndarray,
        blockers: np.ndarray,
    ) -> float | None:
        sx, sy = start.point
        ex, ey = end.point
        vector = np.array([ex - sx, ey - sy], dtype=np.float32)
        distance = float(np.linalg.norm(vector))
        if distance <= 1 or distance > self.max_distance:
            return None

        unit = vector / distance
        start_angle = self._angle_between(start.direction, unit)
        end_angle = self._angle_between(end.direction, -unit)
        if start_angle > self.max_angle or end_angle > self.max_angle:
            return None

        line = self._line_points(start.point, end.point)
        if self._path_crosses_blockers(line, blockers):
            return None

        score = self._template_score(line, road_probability)
        if score < self.min_template_score:
            return None
        return score

    def _template_score(
        self,
        path: list[tuple[int, int]],
        road_probability: np.ndarray,
    ) -> float:
        if not path:
            return 0.0
        values = [float(road_probability[y, x]) for x, y in path]
        return float(np.mean(values))

    def _semantic_cost_map(
        self,
        road_mask: np.ndarray,
        road_probability: np.ndarray,
        semantic_mask: np.ndarray,
        building_class_ids: set[int],
        water_class_ids: set[int],
        rgb: np.ndarray | None,
    ) -> np.ndarray:
        cost = np.full(road_mask.shape, 5.0, dtype=np.float32)
        cost[road_probability >= 0.25] = 2.0
        cost[road_mask > 0] = 1.0

        if rgb is not None:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            cost[(gray < 45) & (road_mask == 0)] = 8.0

        blockers = self._blocker_mask(
            road_mask,
            road_probability,
            semantic_mask,
            building_class_ids,
            water_class_ids,
        )
        cost[blockers] = np.inf
        return cost

    def _blocker_mask(
        self,
        road_mask: np.ndarray,
        road_probability: np.ndarray,
        semantic_mask: np.ndarray,
        building_class_ids: set[int],
        water_class_ids: set[int],
    ) -> np.ndarray:
        water = np.isin(semantic_mask, list(water_class_ids))
        building = np.isin(semantic_mask, list(building_class_ids))
        road_evidence = (road_mask > 0) | (road_probability >= self.min_probability)
        return water | (building & ~road_evidence)

    def _least_cost_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray,
    ) -> list[tuple[int, int]] | None:
        sx, sy = start
        gx, gy = goal
        margin = 8
        x0 = max(0, min(sx, gx) - margin)
        y0 = max(0, min(sy, gy) - margin)
        x1 = min(cost_map.shape[1], max(sx, gx) + margin + 1)
        y1 = min(cost_map.shape[0], max(sy, gy) + margin + 1)

        start_l = (sx - x0, sy - y0)
        goal_l = (gx - x0, gy - y0)
        local_cost = cost_map[y0:y1, x0:x1]

        queue: list[tuple[float, tuple[int, int]]] = [(0.0, start_l)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        best = {start_l: 0.0}

        while queue:
            _, current = heapq.heappop(queue)
            if current == goal_l:
                return self._rebuild_path(came_from, current, x0, y0)

            cx, cy = current
            for nx, ny, step_cost in self._neighbours(cx, cy, local_cost.shape):
                pixel_cost = float(local_cost[ny, nx])
                if not math.isfinite(pixel_cost):
                    continue
                next_cost = best[current] + pixel_cost * step_cost
                neighbour = (nx, ny)
                if next_cost >= best.get(neighbour, math.inf):
                    continue
                best[neighbour] = next_cost
                priority = next_cost + math.hypot(goal_l[0] - nx, goal_l[1] - ny)
                heapq.heappush(queue, (priority, neighbour))
                came_from[neighbour] = current

        return None

    def _validate_path(
        self,
        path: list[tuple[int, int]],
        start: RoadEndpoint,
        end: RoadEndpoint,
        road_probability: np.ndarray,
        cost_map: np.ndarray,
    ) -> bool:
        distance = math.hypot(end.point[0] - start.point[0], end.point[1] - start.point[1])
        if distance > self.max_distance:
            return False

        probabilities = [float(road_probability[y, x]) for x, y in path]
        if float(np.mean(probabilities)) < self.min_probability:
            return False

        costs = [float(cost_map[y, x]) for x, y in path]
        if not costs or float(np.mean(costs)) > self.max_path_cost:
            return False

        if not self._similar_width(start.radius, end.radius):
            return False

        return self._max_curvature(path) <= 45.0

    def _similar_width(self, radius_a: float, radius_b: float) -> bool:
        widest = max(radius_a, radius_b, 1.0)
        return abs(radius_a - radius_b) / widest <= self.max_width_difference

    def _max_curvature(self, path: list[tuple[int, int]], step: int = 5) -> float:
        if len(path) < step * 2 + 1:
            return 0.0

        max_angle = 0.0
        for idx in range(step, len(path) - step):
            prev_pt = np.array(path[idx - step], dtype=np.float32)
            mid_pt = np.array(path[idx], dtype=np.float32)
            next_pt = np.array(path[idx + step], dtype=np.float32)
            v1 = mid_pt - prev_pt
            v2 = next_pt - mid_pt
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            max_angle = max(max_angle, self._angle_between(v1, v2))
        return max_angle

    def _connection_radius(self, start: RoadEndpoint, end: RoadEndpoint) -> int:
        radius = int(round((start.radius + end.radius) / 2.0))
        return max(1, min(radius, 8))

    def _draw_path(
        self,
        mask: np.ndarray,
        path: list[tuple[int, int]],
        radius: int,
    ) -> None:
        for x, y in path:
            cv2.circle(mask, (int(x), int(y)), radius, 1, thickness=-1)

    def _path_crosses_blockers(
        self,
        path: list[tuple[int, int]],
        blockers: np.ndarray,
    ) -> bool:
        return any(blockers[y, x] for x, y in path)

    def _line_points(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[int, int]]:
        sx, sy = start
        ex, ey = end
        steps = max(abs(ex - sx), abs(ey - sy), 1)
        xs = np.linspace(sx, ex, steps + 1)
        ys = np.linspace(sy, ey, steps + 1)
        return [
            (int(round(x)), int(round(y)))
            for x, y in zip(xs, ys)
        ]

    def _rebuild_path(
        self,
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
        x_offset: int,
        y_offset: int,
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [(x + x_offset, y + y_offset) for x, y in path]

    def _neighbours(
        self,
        x: int,
        y: int,
        shape: tuple[int, int],
    ):
        height, width = shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    yield nx, ny, math.sqrt(2.0) if dx and dy else 1.0

    def _angle_between(self, first, second) -> float:
        a = np.asarray(first, dtype=np.float32)
        b = np.asarray(second, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-6:
            return 180.0
        cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        return math.degrees(math.acos(cosine))
