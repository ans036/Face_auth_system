import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

PAIRS_FILE = "evaluation/pairs.txt"
EMB_FILE = "evaluation/embeddings.npy"

emb_map = np.load(EMB_FILE, allow_pickle=True).item()
y_true = []
y_scores = []

with open(PAIRS_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        p1, p2, label = parts
        if p1 not in emb_map or p2 not in emb_map:
            continue
        s = cosine_similarity([emb_map[p1]], [emb_map[p2]])[0][0]
        y_true.append(int(label))
        y_scores.append(float(s))

os.makedirs("evaluation/output", exist_ok=True)
np.save("evaluation/output/y_true.npy", y_true)
np.save("evaluation/output/y_scores.npy", y_scores)
print("Saved evaluation arrays to evaluation/output/")
