from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon


class Zone:
    """Polygon zone for intrusion detection."""

    def __init__(self, name: str, polygon: List[List[int]], color: List[int]):
        """
        Initialize zone.

        Args:
            name: Zone name
            polygon: List of [x, y] coordinates
            color: BGR color for visualization
        """
        self.name = name
        self.polygon_points = np.array(polygon, dtype=np.int32)
        self.color = tuple(color)
        self.polygon = Polygon(polygon)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside zone."""
        return self.polygon.contains(Point(point))

    def draw(self, image: np.ndarray, opacity: float = 0.3) -> np.ndarray:
        """Draw zone on image with transparency."""
        overlay = image.copy()
        cv2.fillPoly(overlay, [self.polygon_points], self.color)
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

        # Draw border
        cv2.polylines(image, [self.polygon_points], True, self.color, 2)

        # Draw zone name
        centroid = self.polygon.centroid
        text_pos = (int(centroid.x), int(centroid.y))
        cv2.putText(image, self.name, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, self.color, 2)

        return image
