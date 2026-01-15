# backend/core/detector.py
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class FaceDetector:
    def __init__(self, min_detection_confidence=0.6):
        # Ensure this path is correct in Docker
        model_path = '/app/models/detector.tflite' 

        base_options = python.BaseOptions(model_asset_path=model_path)
        # ENABLE LANDMARKS
        options = vision.FaceDetectorOptions(
            base_options=base_options, 
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image):
        """
        IMPORTANT: Input image MUST be in RGB format (not BGR)!
        Returns: box (y1, x1, y2, x2), keypoints (for alignment)
        """
        # MediaPipe expects RGB - the image should already be converted by utils.image.read_image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return None, None

        # Get top detection
        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        
        h, w, _ = image.shape
        
        # Convert xywh to y1, x1, y2, x2
        x1 = bbox.origin_x
        y1 = bbox.origin_y
        x2 = x1 + bbox.width
        y2 = y1 + bbox.height

        # Ensure within bounds
        box = [max(0, y1), max(0, x1), min(h, y2), min(w, x2)]
        
        # Get keypoints for alignment (Available in detection.keypoints)
        keypoints = detection.keypoints

        return box, keypoints