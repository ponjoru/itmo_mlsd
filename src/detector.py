from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """Single object detection."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class ONNXYOLODetector:
    """YOLO detector using ONNX Runtime for inference."""

    def __init__(self, model_path: str, imgsz: int = 640,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 target_classes: Optional[List[int]] = None):
        """
        Initialize ONNX YOLO detector.

        Args:
            model_path: Path to ONNX model file
            imgsz: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            target_classes: List of class IDs to keep (None = all)
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes

        # Load COCO class names
        self.class_names = self._load_coco_names()

        # Load model
        self.session = self._load_model()

    def _load_coco_names(self) -> List[str]:
        """Load COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def _load_model(self):
        """Load ONNX model with ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")

        # Check if model exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Create inference session
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(self.model_path, providers=providers)

        print(f"Loaded ONNX model from: {self.model_path}")
        print(f"Input name: {session.get_inputs()[0].name}")
        print(f"Input shape: {session.get_inputs()[0].shape}")
        print(f"Output name: {session.get_outputs()[0].name}")

        return session

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO model.

        Args:
            image: Input image (BGR)

        Returns:
            Preprocessed image, scale ratio, and padding
        """
        # Get original shape
        img_h, img_w = image.shape[:2]

        # Resize with aspect ratio preservation
        scale = min(self.imgsz / img_w, self.imgsz / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_w = self.imgsz - new_w
        pad_h = self.imgsz - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert to RGB and normalize
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0

        # Transpose to CHW format
        padded = np.transpose(padded, (2, 0, 1))

        # Add batch dimension
        padded = np.expand_dims(padded, axis=0)

        return padded, scale, (left, top)

    def postprocess(self, outputs: np.ndarray, scale: float, padding: Tuple[int, int],
                    original_shape: Tuple[int, int]) -> List[Detection]:
        """
        Postprocess YOLO outputs.

        Args:
            outputs: Model outputs
            scale: Scale factor used in preprocessing
            padding: Padding (left, top)
            original_shape: Original image shape (h, w)

        Returns:
            List of detections
        """
        # YOLOv11 output format: [batch, 84, num_detections]
        # 84 = 4 (bbox) + 80 (classes for COCO)
        predictions = outputs[0]  # [84, num_detections]

        # Transpose to [num_detections, 84]
        predictions = predictions.T  # Now [num_detections, 84]

        detections = []

        for pred in predictions:
            # Get bbox and scores
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]

            # Get best class
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            # Filter by confidence
            if confidence < self.conf_threshold:
                continue

            # Filter by target classes
            if self.target_classes is not None and class_id not in self.target_classes:
                continue

            # Convert to x1, y1, x2, y2 format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Adjust for padding
            x1 = (x1 - padding[0]) / scale
            y1 = (y1 - padding[1]) / scale
            x2 = (x2 - padding[0]) / scale
            y2 = (y2 - padding[1]) / scale

            # Clip to image bounds
            img_h, img_w = original_shape
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            bbox = np.array([x1, y1, x2, y2])

            detections.append(Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=self.class_names[class_id]
            ))

        # Apply NMS
        detections = self._nms(detections)

        return detections

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if len(detections) == 0:
            return []

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        # Compute areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by confidence
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on image.

        Args:
            image: Input image (BGR)

        Returns:
            List of detections
        """
        # Preprocess
        input_tensor, scale, padding = self.preprocess(image)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        # Postprocess
        detections = self.postprocess(
            outputs[0], scale, padding, image.shape[:2]
        )

        return detections
