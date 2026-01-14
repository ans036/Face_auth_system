from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.recognizer import Recognizer
from services.audit_logger import log_security_event
from utils.image import read_image, crop_box
from config_loader import load_config
import numpy as np
import os

router = APIRouter()
detector = FaceDetector()
embedder = FaceEmbedder()
recognizer = Recognizer()
config = load_config()

UNLOCK_FILE = config.get("unlock_file", "unlocked.txt")
SEC_LOG = config.get("logging", {}).get("security_log", "security.log")

@router.post("/")
async def identify(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = read_image(content)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        boxes = detector.detect(img)  # list of (xmin, ymin, w, h) normalized

        if not boxes:
            # nothing detected
            log_security_event(None, False, 0.0, "no_face_detected")
            return {"unlocked": False, "warning": "no_face_detected"}

        # process first detected face (you can iterate for multiple)
        x, y, w, h = boxes[0]
        face = crop_box(img, x, y, w, h)
        emb = embedder.embed(face)
        emb = np.asarray(emb, dtype=np.float32)

        match_name, score = recognizer.identify(emb)
        if match_name:
            # Unlock action: create a file demonstrating unlock
            with open(UNLOCK_FILE, "w") as f:
                f.write(f"UNLOCKED_BY:{match_name}\nSCORE:{score}\n")
            log_security_event(match_name, True, float(score), "access_granted")
            return {"unlocked": True, "name": match_name, "score": float(score)}
        else:
            log_security_event(None, False, float(score), "unknown_person_access_denied")
            return {"unlocked": False, "warning": "unknown person", "score": float(score)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
