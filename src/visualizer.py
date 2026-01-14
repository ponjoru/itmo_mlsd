from collections import defaultdict, deque
from typing import Dict, List

import cv2
import numpy as np

from event_generator import EventGenerator
from tracker import Track
from zone_manager import Zone


class VideoVisualizer:
    """Visualize tracks, zones, and statistics on video frames."""

    def __init__(self, config: Dict, zones: List[Zone], event_generator: EventGenerator):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
            zones: List of zones to draw
            event_generator: Event generator for statistics
        """
        self.config = config
        self.zones = zones
        self.event_generator = event_generator

        # Track trails for visualization
        self.track_trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config['trail_length'])
        )

    def draw_info_panel(self, image: np.ndarray, frame_id: int, fps: float,
                       num_detections: int, num_tracks: int) -> np.ndarray:
        """Draw info panel with statistics in top-right corner."""
        panel_cfg = self.config['info_panel']
        h, w = image.shape[:2]

        # Panel dimensions
        panel_w = panel_cfg['width']
        panel_h = panel_cfg['height']
        x = w - panel_w - 10
        y = 10

        # Draw panel background
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h),
                     tuple(panel_cfg['background_color']), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Draw border
        cv2.rectangle(image, (x, y), (x + panel_w, y + panel_h), (255, 255, 255), 2)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = panel_cfg['font_scale']
        font_thickness = panel_cfg['font_thickness']
        color = tuple(panel_cfg['text_color'])
        line_height = 25

        text_x = x + 10
        text_y = y + 30

        # Statistics
        texts = [
            f"Frame: {frame_id}",
            f"FPS: {fps:.1f}",
            f"Detections: {num_detections}",
            f"Active Tracks: {num_tracks}",
            "",
            f"TOTAL ALARMS: {self.event_generator.total_alarms}",
        ]

        for zone_name, count in self.event_generator.alarms_by_zone.items():
            texts.append(f"  {zone_name}: {count}")

        for i, text in enumerate(texts):
            cv2.putText(image, text, (text_x, text_y + i * line_height),
                       font, font_scale, color, font_thickness)

        return image

    def draw_alarm_panel(self, image: np.ndarray, num_intrusions: int) -> np.ndarray:
        """Draw alarm message in top-left corner."""
        alarm_text = "!!! INTRUSION DETECTED !!!"
        detail_text = f"{num_intrusions} person(s) in restricted zone(s)"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        # Get text sizes
        (alarm_w, alarm_h), _ = cv2.getTextSize(alarm_text, font, font_scale, font_thickness)
        (detail_w, detail_h), _ = cv2.getTextSize(detail_text, font, 0.5, 1)

        panel_width = max(alarm_w, detail_w) + 40
        panel_height = alarm_h + detail_h + 40

        # Draw background with transparency
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height),
                     (0, 0, 200), -1)  # Red background
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Draw border
        cv2.rectangle(image, (10, 10), (10 + panel_width, 10 + panel_height),
                     (0, 0, 255), 3)  # Bright red border

        # Draw alarm text
        cv2.putText(image, alarm_text, (30, 35 + alarm_h),
                   font, font_scale, (255, 255, 255), font_thickness)

        # Draw detail text
        cv2.putText(image, detail_text, (30, 45 + alarm_h + detail_h),
                   font, 0.5, (255, 255, 255), 1)

        return image

    def draw_track(self, image: np.ndarray, track: Track, in_zone: bool) -> np.ndarray:
        """
        Draw single track on image.

        Args:
            image: Image to draw on
            track: Track to draw
            in_zone: Whether track is in restricted zone

        Returns:
            Image with track drawn
        """
        x1, y1, x2, y2 = map(int, track.bbox)
        center = (int(track.center[0]), int(track.center[1]))

        # Choose color based on zone status
        if in_zone:
            box_color = (203, 192, 255)  # Pink for tracks in restricted zones
            trail_color = (203, 192, 255)
        else:
            box_color = tuple(self.config['box_color'])  # Green for normal tracks
            trail_color = tuple(self.config['box_color'])

        # Update trail
        if self.config['show_trails']:
            self.track_trails[track.track_id].append(center)

            # Draw trail
            if len(self.track_trails[track.track_id]) > 1:
                points = list(self.track_trails[track.track_id])
                for i in range(len(points) - 1):
                    cv2.line(image, points[i], points[i + 1], trail_color, 2)

        # Draw bounding box
        if self.config['show_boxes']:
            cv2.rectangle(image, (x1, y1), (x2, y2),
                        box_color, self.config['box_thickness'])

        # Draw track ID
        if self.config['show_track_ids']:
            if in_zone:
                label = f"ID:{track.track_id} [ALERT]"
            else:
                label = f"ID:{track.track_id}"

            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.config['font_scale'],
                self.config['font_thickness']
            )
            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1),
                        box_color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, self.config['font_scale'],
                       (0, 0, 0), self.config['font_thickness'])

        return image

    def visualize_frame(self, image: np.ndarray, tracks: List[Track],
                       frame_id: int, fps: float) -> np.ndarray:
        """
        Draw tracks, zones, and info panels on frame.

        Args:
            image: Input frame
            tracks: List of tracks to draw
            frame_id: Current frame ID
            fps: Processing FPS

        Returns:
            Annotated frame
        """
        vis_image = image.copy()

        # Draw zones
        if self.config['show_zones']:
            for zone in self.zones:
                vis_image = zone.draw(vis_image, self.config['zone_opacity'])

        # Track tracks in zones for alarm message
        tracks_in_zones = []

        # Draw tracks
        for track in tracks:
            # Check if track is in any zone
            in_zone = False
            for zone in self.zones:
                if zone.contains_point(track.center):
                    in_zone = True
                    tracks_in_zones.append((track.track_id, zone.name))

            # Draw track
            vis_image = self.draw_track(vis_image, track, in_zone)

        # Draw alarm panel if anyone is in a zone
        if tracks_in_zones:
            vis_image = self.draw_alarm_panel(vis_image, len(tracks_in_zones))

        # Draw info panel
        vis_image = self.draw_info_panel(
            vis_image, frame_id, fps, len(tracks), len(tracks)
        )

        return vis_image
