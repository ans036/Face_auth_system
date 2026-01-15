"""
Enhanced Enrollment API with multi-image upload and quality validation.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import numpy as np
import cv2
import os

from core.embedder import FaceEmbedder
from core.detector import FaceDetector
from db.crud import create_gallery_entry, get_gallery_stats, delete_user_from_gallery
from utils.image import read_image
from utils.allignment import align_face

router = APIRouter()
embedder = FaceEmbedder()
detector = FaceDetector()


@router.post("/")
async def enroll_user(
    username: str = Form(..., description="Username for the person being enrolled"),
    files: List[UploadFile] = File(..., description="One or more face images")
):
    """
    Enroll a new user or add images to existing user's gallery.
    
    - **username**: Name of the person (e.g., "Anish", "Sayani")
    - **files**: Multiple face images for enrollment
    
    Returns enrollment statistics including successful and failed image counts.
    """
    if not username or len(username.strip()) == 0:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    
    username = username.strip()
    
    results = {
        "status": "success",
        "username": username,
        "images_received": len(files),
        "faces_enrolled": 0,
        "failed": [],
        "message": ""
    }
    
    for i, file in enumerate(files):
        filename = file.filename or f"image_{i}"
        
        try:
            # Read image
            content = await file.read()
            if len(content) == 0:
                results["failed"].append({"file": filename, "reason": "Empty file"})
                continue
            
            img = read_image(content)
            if img is None:
                results["failed"].append({"file": filename, "reason": "Invalid image format"})
                continue
            
            # Detect face
            box, kps = detector.detect(img)
            if box is None:
                results["failed"].append({"file": filename, "reason": "No face detected"})
                continue
            
            # Align and crop face
            aligned = align_face(img, kps)
            from utils.image import crop_box
            face = crop_box(aligned, box)
            
            # Quality check - ensure face is large enough
            if face.shape[0] < 50 or face.shape[1] < 50:
                results["failed"].append({"file": filename, "reason": "Face too small"})
                continue
            
            # Generate embedding
            emb = embedder.embed(face)
            emb = np.asarray(emb, dtype=np.float32)
            
            # Verify embedding is valid
            norm = np.linalg.norm(emb)
            if norm < 0.5 or np.isnan(norm):
                results["failed"].append({"file": filename, "reason": "Invalid embedding generated"})
                continue
            
            # Save to gallery
            create_gallery_entry(username, emb)
            results["faces_enrolled"] += 1
            
        except Exception as e:
            results["failed"].append({"file": filename, "reason": str(e)})
    
    # Update message
    if results["faces_enrolled"] > 0:
        results["message"] = f"Successfully enrolled {results['faces_enrolled']} face(s) for {username}"
        
        # Hot-reload the recognizer gallery
        try:
            from api.identify import recognizer
            recognizer.reload_gallery()
            results["message"] += ". Gallery reloaded."
        except Exception as e:
            results["message"] += f". Warning: Gallery reload failed - restart may be needed."
    else:
        results["status"] = "failed"
        results["message"] = "No faces could be enrolled"
    
    return results


@router.post("/single")
async def enroll_single_image(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Enroll a single image (legacy endpoint for compatibility).
    """
    return await enroll_user(username=username, files=[file])


@router.get("/save-from-camera")
async def save_enrollment_instructions():
    """
    Returns instructions for enrolling via camera capture.
    """
    return {
        "instructions": [
            "1. Use the frontend camera view to capture your face",
            "2. Save the captured images to the database folder",
            "3. Or use POST /enroll/ with the image files",
            "4. The system will automatically detect and enroll faces"
        ],
        "endpoint": "POST /enroll/",
        "required_fields": {
            "username": "string (form data)",
            "files": "image files (multipart form)"
        }
    }
