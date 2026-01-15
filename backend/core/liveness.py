"""
Liveness Detection Module
Uses Eye Aspect Ratio (EAR) for blink detection and gaze tracking.

Research basis:
- EAR formula from Soukupová & Čech (2016)
- MediaPipe FaceMesh for 468 facial landmarks
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import time

# MediaPipe FaceMesh landmark indices for eyes
# Left eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
# Right eye landmarks  
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Iris landmarks for gaze detection
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Thresholds
EAR_THRESHOLD = 0.20  # Below this = eyes closed
BLINK_CONSECUTIVE_FRAMES = 2  # Frames eyes must be closed
GAZE_THRESHOLD = 0.35  # How centered iris must be (0 = center, 1 = edge)


class LivenessDetector:
    """
    Detects liveness through blink detection and gaze tracking.
    Low-effort design: waits for natural blinks, no prompts needed.
    """
    
    def __init__(self):
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_counter = 0
        self.ear_history = []
        self.last_blink_time = 0
        self.is_looking_at_screen = False
        
        # For tracking blink state
        self.eyes_closed = False
        self.blink_start_frame = 0
        
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Higher EAR = eyes open, Lower EAR = eyes closed
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if h == 0:
            return 0.0
            
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def get_eye_landmarks(self, face_landmarks: list, indices: list) -> np.ndarray:
        """Extract eye landmarks from MediaPipe face landmarks."""
        return np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in indices])
    
    def detect_blink(self, left_ear: float, right_ear: float) -> bool:
        """
        Detect if a blink occurred.
        Returns True when a complete blink is detected (eyes close then open).
        """
        avg_ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(avg_ear)
        
        # Keep only last 30 frames of history
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)
        
        self.frame_counter += 1
        
        # Check if eyes are closed
        if avg_ear < EAR_THRESHOLD:
            if not self.eyes_closed:
                self.eyes_closed = True
                self.blink_start_frame = self.frame_counter
            self.blink_counter += 1
        else:
            # Eyes are open - check if we just finished a blink
            if self.eyes_closed and self.blink_counter >= BLINK_CONSECUTIVE_FRAMES:
                # Valid blink detected!
                self.total_blinks += 1
                self.last_blink_time = time.time()
                self.eyes_closed = False
                self.blink_counter = 0
                return True
            
            self.eyes_closed = False
            self.blink_counter = 0
        
        return False
    
    def calculate_gaze_ratio(self, eye_landmarks: np.ndarray, iris_center: np.ndarray) -> float:
        """
        Calculate how centered the iris is within the eye.
        Returns 0.0 if looking straight, higher values if looking away.
        """
        # Eye center
        eye_center = np.mean(eye_landmarks, axis=0)
        
        # Eye width
        eye_width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if eye_width == 0:
            return 1.0
        
        # Distance from iris center to eye center
        offset = np.linalg.norm(iris_center - eye_center)
        
        # Normalize by eye width
        gaze_ratio = offset / (eye_width / 2)
        
        return min(gaze_ratio, 1.0)
    
    def is_looking_at_camera(self, left_gaze: float, right_gaze: float) -> bool:
        """Check if user is looking at the camera/screen."""
        avg_gaze = (left_gaze + right_gaze) / 2.0
        self.is_looking_at_screen = avg_gaze < GAZE_THRESHOLD
        return self.is_looking_at_screen
    
    def process_frame(self, face_landmarks) -> Dict:
        """
        Process a frame with face landmarks.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            Dict with liveness info:
            - is_live: True if blink detected recently
            - blink_detected: True if blink just occurred
            - looking_at_screen: True if gazing at camera
            - ear_value: Current EAR value
            - total_blinks: Count of blinks in session
        """
        if face_landmarks is None:
            return {
                "is_live": False,
                "blink_detected": False,
                "looking_at_screen": False,
                "ear_value": 0.0,
                "total_blinks": self.total_blinks,
                "time_since_blink": time.time() - self.last_blink_time if self.last_blink_time > 0 else 999
            }
        
        landmarks = face_landmarks.landmark
        
        # Get eye landmarks
        left_eye = self.get_eye_landmarks(landmarks, LEFT_EYE)
        right_eye = self.get_eye_landmarks(landmarks, RIGHT_EYE)
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Detect blink
        blink_detected = self.detect_blink(left_ear, right_ear)
        
        # Check gaze (if iris landmarks available - indices 468-477)
        looking_at_screen = True
        try:
            if len(landmarks) > 477:
                left_iris = np.array([[landmarks[i].x, landmarks[i].y] for i in LEFT_IRIS])
                right_iris = np.array([[landmarks[i].x, landmarks[i].y] for i in RIGHT_IRIS])
                
                left_iris_center = np.mean(left_iris, axis=0)
                right_iris_center = np.mean(right_iris, axis=0)
                
                left_gaze = self.calculate_gaze_ratio(left_eye, left_iris_center)
                right_gaze = self.calculate_gaze_ratio(right_eye, right_iris_center)
                
                looking_at_screen = self.is_looking_at_camera(left_gaze, right_gaze)
        except:
            pass
        
        # Determine if "live" - had a blink within last 5 seconds
        time_since_blink = time.time() - self.last_blink_time if self.last_blink_time > 0 else 999
        is_live = time_since_blink < 5.0 and self.total_blinks > 0
        
        return {
            "is_live": is_live,
            "blink_detected": blink_detected,
            "looking_at_screen": looking_at_screen,
            "ear_value": avg_ear,
            "total_blinks": self.total_blinks,
            "time_since_blink": time_since_blink
        }
    
    def reset(self):
        """Reset the detector state."""
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_counter = 0
        self.ear_history = []
        self.last_blink_time = 0
        self.eyes_closed = False


# Singleton instance for API use
_liveness_detector = None

def get_liveness_detector() -> LivenessDetector:
    """Get or create the liveness detector singleton."""
    global _liveness_detector
    if _liveness_detector is None:
        _liveness_detector = LivenessDetector()
    return _liveness_detector
