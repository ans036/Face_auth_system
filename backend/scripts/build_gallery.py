import os
import cv2
import numpy as np
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from db.crud import create_gallery_entry, clear_gallery, create_voice_entry, clear_voice_gallery
from utils.image import crop_box
from utils.allignment import align_face

# Import quality assessment (place image_quality.py in utils/)
try:
    from utils.image_quality import is_face_quality_acceptable, enhance_face_image
    QUALITY_CHECK_ENABLED = True
except ImportError:
    print("⚠️  Image quality module not found, skipping quality checks")
    QUALITY_CHECK_ENABLED = False

# Import voice embedder (optional - falls back to face-only if unavailable)
VOICE_ENABLED = False
try:
    from core.voice_embedder import get_voice_embedder
    VOICE_ENABLED = True
except Exception as e:
    print(f"⚠️  Voice embedder not available: {e}")
    print("   System will run in face-only mode")

detector = FaceDetector()
embedder = FaceEmbedder()

def apply_clahe(img):
    """Apply CLAHE preprocessing - MUST match what read_image() does!"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    img_clahe = cv2.merge((l_clahe, a, b))
    img_bgr = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
    
    return img_bgr

def build_voice_gallery(database_root="/app/database"):
    """Build voice gallery from audio files in user directories."""
    if not VOICE_ENABLED:
        print("⚠️  Voice embedder not available, skipping voice gallery")
        return
    
    voice_embedder = get_voice_embedder()
    if not voice_embedder.is_available():
        print("⚠️  SpeechBrain model not loaded, skipping voice gallery")
        return
    
    print("🎤 Building voice gallery...")
    clear_voice_gallery()
    
    users = [d for d in os.listdir(database_root) if os.path.isdir(os.path.join(database_root, d))]
    
    for user in users:
        user_dir = os.path.join(database_root, user)
        audio_files = [f for f in os.listdir(user_dir) if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))]
        
        count = 0
        for f in audio_files:
            audio_path = os.path.join(user_dir, f)
            
            try:
                emb = voice_embedder.embed(audio_path)
                if emb is not None:
                    create_voice_entry(user, emb)
                    count += 1
                else:
                    print(f"⚠️  {user}/{f}: Failed to extract voice embedding")
            except Exception as e:
                print(f"⚠️  {user}/{f}: Error processing audio - {e}")
        
        if count > 0:
            print(f"✅ User {user}: Stored {count} voice templates")

def build_gallery(database_root="/app/database", min_quality=0.25):
    """
    Build gallery with quality filtering and enhanced preprocessing.
    
    Args:
        database_root: Root directory containing user folders
        min_quality: Minimum acceptable quality score (0-1)
    """
    print("🧹 Cleaning gallery for fresh build...")
    clear_gallery()
    
    users = [d for d in os.listdir(database_root) if os.path.isdir(os.path.join(database_root, d))]
    
    for user in users:
        user_dir = os.path.join(database_root, user)
        img_files = [f for f in os.listdir(user_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        count = 0
        skipped_quality = 0
        skipped_detection = 0
        
        for f in img_files:
            img_path = os.path.join(user_dir, f)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"⚠️  {user}/{f}: Failed to load, skipping...")
                continue
            
            # Apply CLAHE preprocessing
            img = apply_clahe(img)
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect face
            box, kps = detector.detect(img_rgb)
            if not box:
                skipped_detection += 1
                continue
            
            # Align and crop face
            aligned = align_face(img_rgb, kps)
            face = crop_box(aligned, box)
            
            # Quality check (if enabled)
            if QUALITY_CHECK_ENABLED:
                acceptable, quality_score, reason = is_face_quality_acceptable(face, min_quality)
                if not acceptable:
                    print(f"⚠️  {user}/{f}: Quality too low ({quality_score:.2f}) - {reason}")
                    skipped_quality += 1
                    continue
                
                # Enhance face quality
                face = enhance_face_image(face)
            
            # Generate embedding (embedder now handles normalization)
            emb = embedder.embed(face)
            create_gallery_entry(user, emb.astype(np.float32))
            count += 1
        
        print(f"✅ User {user}: Stored {count} templates (skipped: {skipped_detection} no-face, {skipped_quality} low-quality)")

def build_gallery_on_startup():
    """Build both face and voice galleries on startup."""
    build_gallery("/app/database", min_quality=0.20)
    build_voice_gallery("/app/database")