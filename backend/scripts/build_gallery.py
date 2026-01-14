import os
import cv2
import numpy as np
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from db.crud import create_gallery_entry, clear_gallery
from utils.image import read_image, crop_box
from utils.allignment import align_face

detector = FaceDetector()
embedder = FaceEmbedder()

def build_gallery(database_root="/app/database"):
    print(f"Building Multi-Template Gallery from {database_root}...")
    clear_gallery()
    
    users = [d for d in os.listdir(database_root) if os.path.isdir(os.path.join(database_root, d))]
    for user in users:
        user_dir = os.path.join(database_root, user)
        img_files = [f for f in os.listdir(user_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        success_count = 0
        for f in img_files:
            img_path = os.path.join(user_dir, f)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Detect
            box, kps = detector.detect(img)
            if box is not None:
                # Align (Safe version returns original image if keypoints missing)
                aligned = align_face(img, kps)
                face = crop_box(aligned, box)
                emb = embedder.embed(face)
                create_gallery_entry(user, emb.astype(np.float32))
                success_count += 1
        
        print(f"Stored {success_count} templates for user: {user}")

def build_gallery_on_startup():
    build_gallery("/app/database")