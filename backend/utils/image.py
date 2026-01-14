import numpy as np
import cv2

def read_image(bytestr: bytes):
    arr = np.frombuffer(bytestr, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # Returns BGR
    
    # Lighting Normalization (CLAHE) in BGR color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    
    # CRITICAL FIX: Convert BGR to RGB for MediaPipe and ArcFace
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # Now returns RGB

def crop_box(img, box):
    """Accepts box as [y1, x1, y2, x2] and returns cropped face."""
    y1, x1, y2, x2 = [int(v) for v in box]
    # Ensure coordinates are within image boundaries
    h, w = img.shape[:2]
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = min(h, y2), min(w, x2)
    return img[y1:y2, x1:x2].copy()