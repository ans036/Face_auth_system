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
    print("🧹 Cleaning gallery for fresh build...")
    clear_gallery()
    
    users = [d for d in os.listdir(database_root) if os.path.isdir(os.path.join(database_root, d))]
    for user in users:
        user_dir = os.path.join(database_root, user)
        img_files = [f for f in os.listdir(user_dir) if f.lower().endswith((".jpg", ".png"))]
        
        count = 0
        for f in img_files:
            img = cv2.imread(os.path.join(user_dir, f))
            box, kps = detector.detect(img)
            if box:
                # Use the new zoomed-in alignment
                aligned = align_face(img, kps)
                face = crop_box(aligned, box)
                
                # Only 2 variations: Original and slightly brighter
                variants = [face, cv2.convertScaleAbs(face, alpha=1.1, beta=5)]
                for v in variants:
                    emb = embedder.embed(v)
                    emb = emb / np.linalg.norm(emb)
                    create_gallery_entry(user, emb.astype(np.float32))
                    count += 1
        print(f"✅ User {user}: Stored {count} high-quality templates.")

def build_gallery_on_startup():
    build_gallery("/app/database")