import cv2
import numpy as np

def align_face(image, keypoints):
    """Straightens face based on eyes. Skips if keypoints are missing."""
    if not keypoints or len(keypoints) < 2:
        return image # Return original if no eye data available

    h, w = image.shape[:2]
    
    # MediaPipe: 0=Left Eye, 1=Right Eye
    try:
        left_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
        right_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    except Exception:
        return image