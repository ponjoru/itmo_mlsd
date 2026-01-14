import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List

from tracker import Track
from zone_manager import Zone


@dataclass
class Event:
    """Zone intrusion event."""
    event_type: str  # 'entry', 'exit', 'alarm', 'dwelling'
    track_id: int
    zone_name: str
    timestamp: float
    frame_id: int
    bbox: List[float]
    metadata: Dict = field(default_factory=dict)


class EventGenerator:
    """Generate events based on track-zone interactions."""

    def __init__(self, zones: List[Zone], min_detections_for_alarm: int = 5,
                 alarm_cooldown: int = 30, min_dwelling_time: int = 50,
                 generate_entry: bool = True, generate_exit: bool = True,
                 generate_dwelling: bool = True):
        """
        Initialize event generator.

        Args:
            zones: List of zones to monitor
            min_detections_for_alarm: Min consecutive detections to trigger alarm
            alarm_cooldown: Frames before same track can trigger another alarm
            min_dwelling_time: Min frames in zone to consider as dwelling
            generate_entry: Generate entry events
            generate_exit: Generate exit events
            generate_dwelling: Generate dwelling events
        """
        self.zones = zones
        self.min_detections_for_alarm = min_detections_for_alarm
        self.alarm_cooldown = alarm_cooldown
        self.min_dwelling_time = min_dwelling_time
        self.generate_entry = generate_entry
        self.generate_exit = generate_exit
        self.generate_dwelling = generate_dwelling

        # Track state
        self.track_zone_history: Dict[int, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=100))
        )
        self.track_in_zone: Dict[int, Dict[str, bool]] = defaultdict(
            lambda: defaultdict(bool)
        )
        self.last_alarm_frame: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(lambda: -alarm_cooldown)
        )
        self.zone_entry_frame: Dict[int, Dict[str, int]] = defaultdict(dict)

        # Statistics
        self.total_alarms = 0
        self.alarms_by_zone: Dict[str, int] = defaultdict(int)

    def update(self, tracks: List[Track], frame_id: int) -> List[Event]:
        """
        Update with new tracks and generate events.

        Args:
            tracks: Current frame tracks
            frame_id: Current frame ID

        Returns:
            List of generated events
        """
        events = []

        for track in tracks:
            track_id = track.track_id
            point = track.center

            for zone in self.zones:
                zone_name = zone.name
                in_zone = zone.contains_point(point)

                # Update history
                self.track_zone_history[track_id][zone_name].append(in_zone)

                # Check for entry event
                if in_zone and not self.track_in_zone[track_id][zone_name]:
                    self.track_in_zone[track_id][zone_name] = True
                    self.zone_entry_frame[track_id][zone_name] = frame_id

                    if self.generate_entry:
                        events.append(Event(
                            event_type='entry',
                            track_id=track_id,
                            zone_name=zone_name,
                            timestamp=time.time(),
                            frame_id=frame_id,
                            bbox=track.bbox.tolist(),
                            metadata={'center': point}
                        ))

                # Check for exit event
                elif not in_zone and self.track_in_zone[track_id][zone_name]:
                    self.track_in_zone[track_id][zone_name] = False

                    # Calculate dwelling time
                    if zone_name in self.zone_entry_frame[track_id]:
                        entry_frame = self.zone_entry_frame[track_id][zone_name]
                        dwelling_time = frame_id - entry_frame

                        if self.generate_exit:
                            events.append(Event(
                                event_type='exit',
                                track_id=track_id,
                                zone_name=zone_name,
                                timestamp=time.time(),
                                frame_id=frame_id,
                                bbox=track.bbox.tolist(),
                                metadata={
                                    'center': point,
                                    'dwelling_time': dwelling_time
                                }
                            ))

                        # Check for dwelling event
                        if self.generate_dwelling and dwelling_time >= self.min_dwelling_time:
                            events.append(Event(
                                event_type='dwelling',
                                track_id=track_id,
                                zone_name=zone_name,
                                timestamp=time.time(),
                                frame_id=frame_id,
                                bbox=track.bbox.tolist(),
                                metadata={
                                    'center': point,
                                    'dwelling_time': dwelling_time
                                }
                            ))

                # Check for alarm event (k consecutive detections in zone)
                if in_zone:
                    history = self.track_zone_history[track_id][zone_name]
                    consecutive_in_zone = sum(1 for x in list(history)[-self.min_detections_for_alarm:] if x)

                    if consecutive_in_zone >= self.min_detections_for_alarm:
                        # Check cooldown
                        last_alarm = self.last_alarm_frame[track_id][zone_name]
                        if frame_id - last_alarm >= self.alarm_cooldown:
                            self.last_alarm_frame[track_id][zone_name] = frame_id
                            self.total_alarms += 1
                            self.alarms_by_zone[zone_name] += 1

                            events.append(Event(
                                event_type='alarm',
                                track_id=track_id,
                                zone_name=zone_name,
                                timestamp=time.time(),
                                frame_id=frame_id,
                                bbox=track.bbox.tolist(),
                                metadata={
                                    'center': point,
                                    'consecutive_detections': consecutive_in_zone
                                }
                            ))

        return events
