import os
import cv2
import numpy as np
from core.embedder import FaceEmbedder
from db.crud import create_gallery_entry, clear_gallery
from config_loader import load_config
from tqdm import tqdm

def build_gallery(database_root = "/app/database"):
    """
    Scans database_root/<username>/*.jpg and builds averaged embeddings for each username.
    """
    embedder = FaceEmbedder()
    users = [d for d in os.listdir(database_root) if os.path.isdir(os.path.join(database_root, d))]
    clear_gallery()
    for user in users:
        user_dir = os.path.join(database_root, user)
        images = [os.path.join(user_dir, f) for f in os.listdir(user_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not images:
            continue
        embs = []
        for p in images:
            img = cv2.imread(p)
            if img is None:
                print(f"Warning: could not read {p}")
                continue
            emb = embedder.embed(img)
            embs.append(emb)
        if not embs:
            continue
        embs = np.vstack(embs)
        avg_emb = np.mean(embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        create_gallery_entry(user, avg_emb.astype(np.float32))
        print(f"Added gallery entry for {user} ({len(embs)} images)")
    print("Gallery build complete.")

def build_gallery_on_startup():
    # Force the path to match the Docker volume mount point
    database_root = "/app/database" 
    if os.path.exists(database_root):
        print(f"Scanning gallery at: {database_root}")
        build_gallery(database_root)
    else:
        print("Error: Gallery folder not found at", database_root)
