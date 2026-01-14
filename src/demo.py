"""
Perimeter Security Human Zone Intrusion Demo

This script processes videos to detect humans entering restricted zones using:
- YOLOv11n ONNX for object detection
- ByteTrack for multi-object tracking
- Zone intrusion detection with event generation
- Video visualization with tracks and statistics
"""

import argparse

import yaml

from video_processor import VideoProcessor


def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description='Human detection and zone intrusion demo'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save output video (default: <video_name>_output.mp4)')

    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override output path if specified
    if args.save_path:
        config['output']['output_path'] = args.save_path

    # Create processor
    processor = VideoProcessor(config)

    # Process video
    processor.process_video(args.video)

    print("\nDone!")


if __name__ == '__main__':
    main()
