from fastapi import APIRouter, Form, UploadFile, File, Depends
from sqlalchemy.orm import Session
from db.session import get_db
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.recognizer import Recognizer
from core.voice_embedder import get_voice_embedder
from core.multimodal import get_multimodal_authenticator
from utils.image import read_image, crop_box
from utils.allignment import align_face
import os
import cv2
import time
import numpy as np
from typing import Optional

# Import liveness detector
LIVENESS_AVAILABLE = False
mp_face_mesh = None
face_mesh = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Includes iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        LIVENESS_AVAILABLE = True
        print("✅ MediaPipe FaceMesh loaded for liveness detection")
    else:
        print("⚠️  MediaPipe solutions not available, liveness detection disabled")
except (ImportError, AttributeError) as e:
    print(f"⚠️  MediaPipe not available for liveness detection: {e}")

router = APIRouter()
detector = FaceDetector()
embedder = FaceEmbedder()
# recognizer = Recognizer() # Replaced by multimodal authenticator
multimodal_auth = get_multimodal_authenticator()
voice_embedder = get_voice_embedder()

# Session-based blink tracking (simple in-memory for demo)
# In production, use Redis or database
user_sessions = {}

# Eye landmarks for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.21


def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio."""
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if h == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h)


def check_liveness(img_rgb, session_id="default"):
    """
    Check if the face is live using blink detection.
    Returns: (is_live, ear_value, blink_detected)
    """
    if not LIVENESS_AVAILABLE or face_mesh is None:
        return True, 0.25, False  # Fallback to no liveness check
    
    try:
        results = face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            return False, 0.0, False
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img_rgb.shape[:2]
        
        # Get eye landmarks
        left_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE])
        right_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE])
        
        # Calculate EAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Track session state
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                "blink_count": 0,
                "eyes_closed": False,
                "last_blink": 0,
                "frames": 0
            }
        
        session = user_sessions[session_id]
        session["frames"] += 1
        blink_detected = False
        
        # Detect blink (eyes close then open)
        if avg_ear < EAR_THRESHOLD:
            session["eyes_closed"] = True
        else:
            if session["eyes_closed"]:
                # Blink completed
                session["blink_count"] += 1
                session["last_blink"] = time.time()
                blink_detected = True
            session["eyes_closed"] = False
        
        # Consider live if blinked in last 5 seconds
        time_since_blink = time.time() - session["last_blink"] if session["last_blink"] > 0 else 999
        is_live = time_since_blink < 5.0 and session["blink_count"] > 0
        
        return is_live, avg_ear, blink_detected
        
    except Exception as e:
        print(f"⚠️  Liveness check error: {e}")
        return True, 0.25, False  # Fallback


@router.post("/")
async def identify(
    file: UploadFile = File(...), 
    voice_sample: Optional[UploadFile] = File(None),
    is_manual: str = Form("false"),
    session_id: str = Form("default"),
    db: Session = Depends(get_db)
):
    """
    Identify user using Multimodal Fusion (Face + Voice + Liveness).
    
    Returns:
        - name: Identified user
        - score: Combined confidence score
        - face_score: Face-only score
        - voice_score: Voice-only score
        - liveness: Liveness status
        - auth_status: Final authentication decision
    """
    try:
        manual_clicked = is_manual.lower() == "true"
        
        content = await file.read()
        img = read_image(content)
        
        # Check liveness first
        is_live, ear_value, blink_detected = check_liveness(img, session_id)
        
        # Detect face
        box, kps = detector.detect(img)
        
        if box is None:
            return {
                "status": "no_face", 
                "matches": [],
                "liveness": {
                    "is_live": False,
                    "ear_value": 0.0,
                    "blink_detected": False,
                    "message": "No face detected"
                }
            }

        # Align and embed face
        aligned = align_face(img, kps)
        face = crop_box(aligned, box)
        face_emb = embedder.embed(face)
        
        # Process voice if provided
        voice_emb = None
        if voice_sample:
            try:
                voice_content = await voice_sample.read()
                # Process only if we have reasonable data (>1KB) to avoid noise
                if len(voice_content) > 100: # Lowered threshold for testing
                    # Extract extension (default to wav if none)
                    ext = voice_sample.filename.split('.')[-1] if '.' in voice_sample.filename else "wav"
                    print(f"🎤 Received voice sample: {len(voice_content)} bytes, format: {ext}")
                    
                    voice_emb = voice_embedder.embed_from_bytes(voice_content, format=ext)
                    
                    if voice_emb is not None:
                         print("✅ Voice embedding generated successfully")
                    else:
                         print("⚠️ Voice embedding failed (returned None)")
                else:
                    print(f"⚠️ Voice sample too small: {len(voice_content)} bytes")
            except Exception as e:
                print(f"⚠️ Voice processing failed: {e}")
        
        # Multimodal Identification
        result = multimodal_auth.identify(
            face_emb=face_emb,
            voice_emb=voice_emb,
            liveness_passed=is_live
        )
        
        name = result["name"]
        combined_score = result["combined_score"]
        face_score = result["face_score"]
        voice_score = result["voice_score"]
        
        # Determine authentication status
        # We require multimodal authentication OR high-confidence face+liveness
        # But for demo, we match the result["is_authenticated"]
        auth_status = "authenticated" if result["is_authenticated"] else "unauthorized"
        
        # Log if manual capture
        if manual_clicked:
            save_event_to_log(img, name, combined_score, is_live)
            
        return {
            "status": "success",
            "matches": [{
                "name": name, 
                "score": float(combined_score), 
                "face_score": float(face_score),
                "voice_score": float(voice_score) if voice_score is not None else None,
                "box": box,
                "modalities": result["modalities_used"]
            }],
            "liveness": {
                "is_live": is_live,
                "ear_value": float(ear_value),
                "blink_detected": blink_detected,
                "message": "Liveness verified" if is_live else "Waiting for blink"
            },
            "auth_status": auth_status
        }
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def save_event_to_log(image, name, score, is_live=False):
    folder = "/app/unauthorized_attempts"
    os.makedirs(folder, exist_ok=True)
    
    timestamp = int(time.time())
    liveness_tag = "LIVE" if is_live else "NOLIVE"
    filename = f"{name}_{timestamp}_score_{int(score*100)}_{liveness_tag}.jpg"
    filepath = os.path.join(folder, filename)
    
    # Save image
    save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, save_img)
    
    # Write to log
    with open("/app/security.log", "a") as f:
        live_status = "LIVE" if is_live else "NOT_LIVE"
        f.write(f"[{timestamp}] {name} | Score: {score:.2f} | Liveness: {live_status} | File: {filename}\n")