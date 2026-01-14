from fastapi import APIRouter, Form, UploadFile, File, Depends
from sqlalchemy.orm import Session
from db.session import get_db
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.recognizer import Recognizer
from utils.image import read_image, crop_box
from utils.allignment import align_face
import os
import cv2
import time

router = APIRouter()
detector = FaceDetector()
embedder = FaceEmbedder()
recognizer = Recognizer()

@router.post("/")
async def identify(
    file: UploadFile = File(...), 
    is_manual: str = Form("false"), # Received as string from FormData
    db: Session = Depends(get_db)
):
    try:
        # Convert string to boolean safely
        manual_clicked = is_manual.lower() == "true"
        
        content = await file.read()
        img = read_image(content)
        box, kps = detector.detect(img)
        
        if box is None:
            return {"status": "no_face", "matches": []}

        aligned = align_face(img, kps)
        face = crop_box(aligned, box)
        emb = embedder.embed(face)
        name, score = recognizer.identify(emb)
        
        # LOGGING TRIGGER: Now robustly checks the manual click
        if manual_clicked:
            save_event_to_log(img, name, score)
            
        return {
            "status": "success",
            "matches": [{"name": name, "score": float(score), "box": box}]
        }
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"status": "error", "message": str(e)}

def save_event_to_log(image, name, score):
    folder = "/app/unauthorized_attempts"
    os.makedirs(folder, exist_ok=True)
    
    timestamp = int(time.time())
    # Filename format: name_timestamp_score_SCORE.jpg
    filename = f"{name}_{timestamp}_score_{int(score*100)}.jpg"
    filepath = os.path.join(folder, filename)
    
    # Save image (Convert RGB to BGR for OpenCV)
    save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, save_img)
    
    # Write to text log
    with open("/app/security.log", "a") as f:
        f.write(f"[{timestamp}] MANUAL CAPTURE: {name} | Score: {score:.4f} | File: {filename}\n")
        f.flush()# Force write to disk
