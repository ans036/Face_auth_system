import os
import cv2
import numpy as np
from tqdm import tqdm
from backend.core.embedder import FaceEmbedder

DATASET_DIR = "evaluation/lfw"
OUTPUT = "evaluation/embeddings.npy"

embedder = FaceEmbedder()
embeddings = {}

for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(root, f)
        img = cv2.imread(path)
        if img is None:
            continue
        emb = embedder.embed(img)
        embeddings[path] = emb

np.save(OUTPUT, embeddings)
print(f"Saved embeddings for {len(embeddings)} images to {OUTPUT}")
