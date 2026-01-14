import numpy as np
from db.crud import get_all_gallery
from sklearn.metrics.pairwise import cosine_similarity
import os

THRESHOLD = float(os.environ.get("IDENTIFICATION_THRESHOLD", 0.15))

class Recognizer:
    def __init__(self, threshold=THRESHOLD):
        self.threshold = float(threshold)

    def identify(self, probe_embedding: np.ndarray):
        """
        Returns (username, score) or (None, best_score)
        """
        gallery = get_all_gallery()  # returns list of dicts: {"username":..., "embedding": np.array}
        if not gallery:
            return None, 0.0

        names = [g["username"] for g in gallery]
        embs = [g["embedding"] for g in gallery]
        embs = np.vstack(embs)
        scores = cosine_similarity([probe_embedding], embs)[0]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= self.threshold:
            return names[best_idx], best_score
        else:
            return None, best_score
