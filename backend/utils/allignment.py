import cv2
import numpy as np

def align_face(image, keypoints):
    """Aligns and crops tightly to internal facial features to reduce background noise."""
    if not keypoints or len(keypoints) < 2:
        return image

    h, w = image.shape[:2]
    left_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
    right_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))

    # Calculate rotation
    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Pivot around center of eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    # Increase scale slightly to 'zoom in' on the face and remove hair/background
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.2)
    aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return aligned