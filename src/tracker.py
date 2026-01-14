from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from detector import Detection


@dataclass
class Track:
    """Single track with ID and history."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        self.center = (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )


class ByteTrackerLite:
    """
    Lightweight ByteTrack implementation without PyTorch dependency.
    Source: https://github.com/kadirnar/bytetrack-pip
    """

    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, min_box_area: float = 10,
                 frame_rate: int = 30):
        try:
            from bytetracker import BYTETracker
        except ImportError:
            raise ImportError(
                "Please install bytetrack-pip: pip install bytetrack\n"
                "This is a lightweight PyTorch-free implementation.\n"
                "Source: https://github.com/kadirnar/bytetrack-pip"
            )

        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate
        )

        self.min_box_area = min_box_area
        self.frame_id = 0

    def _fix_bbox_coords(self, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w, h = x2-x1, y2-y1
        x1 = x1 + w / 2
        y1 = y1 + h / 2
        x2 = x2 + w / 2
        y2 = y2 + h / 2
        return np.array([x1, y1, x2, y2])

    def update(self, detections: List[Detection], image: np.ndarray) -> List[Track]:
        self.frame_id += 1

        if len(detections) == 0:
            # Update with empty detections
            online_targets = self.tracker.update(dets=np.empty((0, 6)))
        else:
            # Convert detections to ByteTrack format: [x1, y1, x2, y2, score]
            dets = np.array([
                [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_id]
                for d in detections
            ])

            # Update tracker
            online_targets = self.tracker.update(dets=dets)

        # Convert to Track objects
        tracks = []

        if online_targets is not None and len(online_targets) > 0:
            for target in online_targets:
                bbox = self._fix_bbox_coords(target[:4])

                track_id = int(target[4])
                score = target[5] if len(target) > 5 else 1.0
                class_id = int(target[6]) if len(target) > 6 else 0

                box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                if box_area < self.min_box_area:
                    continue

                # Get class ID (for person detection, assume class_id = 0)
                class_id = 0

                tracks.append(Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=score,
                    class_id=class_id,
                    class_name='person'
                ))

        return tracks
