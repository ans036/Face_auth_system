from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from db.session import get_db
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.recognizer import Recognizer
from utils.image import read_image, crop_box
from utils.allignment import align_face
import os
import cv2

router = APIRouter()
detector = FaceDetector()
embedder = FaceEmbedder()
recognizer = Recognizer()

@router.post("/")
async def identify(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        img = read_image(await file.read())
        box, kps = detector.detect(img)

        if box is None:
            return {"status": "no_face", "matches": []}

        aligned = align_face(img, kps)
        face = crop_box(aligned, box)
        emb = embedder.embed(face)
        
        # Get result from the stricter recognizer
        name, score = recognizer.identify(emb)
        
        # If score is very low (e.g., < 0.3), we don't even want to show a name.
        # This prevents your friend from being called "Anish (31%)"
        is_identified = name != "Unknown" and score >= 0.6

        return {
            "status": "success",
            "matches": [{
                "name": name if is_identified else "Unknown",
                "score": float(score),
                "box": box
            }]
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}

def save_unauthorized_snapshot(image, score):
    """Saves a timestamped image of the unauthorized person."""
    folder = "/app/unauthorized_attempts"
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    timestamp = int(time.time())
    filename = f"unknown_{timestamp}_score_{int(score*100)}.jpg"
    filepath = os.path.join(folder, filename)
    
    # Convert RGB back to BGR for OpenCV saving
    save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, save_img)
    
    # Log to security audit file
    with open("/app/security.log", "a") as f:
        f.write(f"[{timestamp}] ALERT: Unauthorized attempt detected. Score: {score:.2f}. Image saved to {filename}\n")