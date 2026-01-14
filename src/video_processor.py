import json
import time
from pathlib import Path
from typing import Dict, List

import cv2

from detector import ONNXYOLODetector
from event_generator import Event, EventGenerator
from tracker import ByteTrackerLite
from visualizer import VideoVisualizer
from zone_manager import Zone


class VideoProcessor:
    """Process video with detection, tracking, and visualization."""

    def __init__(self, config: Dict):
        """
        Initialize video processor with config.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize detector
        print("Loading detector...")
        self.detector = ONNXYOLODetector(
            model_path=config['model']['weights'],
            imgsz=config['model']['imgsz'],
            conf_threshold=config['model']['conf_threshold'],
            iou_threshold=config['model']['iou_threshold'],
            target_classes=config['model']['target_classes']
        )

        # Initialize tracker
        print("Initializing tracker...")
        self.tracker = ByteTrackerLite(
            track_thresh=config['tracker']['track_thresh'],
            track_buffer=config['tracker']['track_buffer'],
            match_thresh=config['tracker']['match_thresh'],
            min_box_area=config['tracker']['min_box_area']
        )

        # Initialize zones
        print("Setting up zones...")
        self.zones = [
            Zone(z['name'], z['polygon'], z['color'])
            for z in config['zones']
        ]

        # Initialize event generator
        print("Initializing event generator...")
        event_cfg = config['events']
        self.event_generator = EventGenerator(
            zones=self.zones,
            min_detections_for_alarm=event_cfg['min_detections_for_alarm'],
            alarm_cooldown=event_cfg['alarm_cooldown'],
            min_dwelling_time=event_cfg['min_dwelling_time'],
            generate_entry=event_cfg['generate_entry_events'],
            generate_exit=event_cfg['generate_exit_events'],
            generate_dwelling=event_cfg['generate_dwelling_events']
        )

        # Initialize visualizer
        print("Initializing visualizer...")
        self.visualizer = VideoVisualizer(
            config=config['visualization'],
            zones=self.zones,
            event_generator=self.event_generator
        )

        # All events
        self.all_events: List[Event] = []

    def process_video(self, video_path: str) -> None:
        """
        Process video and generate output.

        Args:
            video_path: Path to input video
        """
        print(f"\nProcessing video: {video_path}")
        print("=" * 80)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")

        # Setup output
        output_cfg = self.config['output']
        out_writer = None

        if output_cfg['save_video']:
            output_path = output_cfg['output_path']
            if output_path is None:
                video_name = Path(video_path).stem
                output_path = f"{video_name}_output.mp4"

            fourcc = cv2.VideoWriter_fourcc(*output_cfg['codec'])
            out_fps = output_cfg['fps'] if output_cfg['fps'] else fps
            out_writer = cv2.VideoWriter(
                output_path, fourcc, out_fps, (frame_width, frame_height)
            )
            print(f"Output video: {output_path}")

        # Process frames
        frame_id = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Detect
            detections = self.detector.detect(frame)

            # Track
            tracks = self.tracker.update(detections, frame)

            # Generate events
            events = self.event_generator.update(tracks, frame_id)
            self.all_events.extend(events)

            # Print events
            for event in events:
                print(f"[Frame {frame_id}] {event.event_type.upper()}: "
                      f"Track {event.track_id} in {event.zone_name}")

            # Visualize
            current_fps = frame_id / (time.time() - start_time)
            vis_frame = self.visualizer.visualize_frame(frame, tracks, frame_id, current_fps)

            # Write output
            if out_writer:
                out_writer.write(vis_frame)

            # Progress
            if frame_id % 30 == 0:
                print(f"Processed {frame_id}/{total_frames} frames "
                      f"({frame_id/total_frames*100:.1f}%) - {current_fps:.1f} FPS")

        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()

        elapsed = time.time() - start_time
        print("\nProcessing complete!")
        print(f"Total frames: {frame_id}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average FPS: {frame_id/elapsed:.2f}")
        print(f"Total events: {len(self.all_events)}")
        print(f"Total alarms: {self.event_generator.total_alarms}")

        # Save events
        if output_cfg['save_events']:
            events_path = output_cfg['events_path']
            if events_path is None:
                video_name = Path(video_path).stem
                events_path = f"{video_name}_events.json"

            events_data = [
                {
                    'event_type': e.event_type,
                    'track_id': e.track_id,
                    'zone_name': e.zone_name,
                    'timestamp': e.timestamp,
                    'frame_id': e.frame_id,
                    'bbox': e.bbox,
                    'metadata': e.metadata
                }
                for e in self.all_events
            ]

            with open(events_path, 'w') as f:
                json.dump(events_data, f, indent=2)

            print(f"Events saved to: {events_path}")
