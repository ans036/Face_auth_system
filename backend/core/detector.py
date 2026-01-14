import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self, min_detection_confidence=0.6):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        boxes = []
        if results.detections:
            h, w, _ = frame.shape
            for d in results.detections:
                b = d.location_data.relative_bounding_box
                boxes.append((b.xmin, b.ymin, b.width, b.height))
        return boxes
