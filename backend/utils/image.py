import numpy as np
import cv2

def read_image(bytestr: bytes):
    arr = np.frombuffer(bytestr, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def crop_box(img, xmin, ymin, w, h):
    # xmin etc are relative (0-1)
    H, W = img.shape[:2]
    x1 = max(0, int(xmin * W))
    y1 = max(0, int(ymin * H))
    x2 = min(W, int((xmin + w) * W))
    y2 = min(H, int((ymin + h) * H))
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img[y1:y2, x1:x2].copy()
