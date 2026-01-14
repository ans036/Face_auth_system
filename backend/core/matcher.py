import numpy as np
from db.crud import get_all_gallery

class FaceMatcher:
    def __init__(self):
        self.gallery = []
        self.load_gallery()

    def load_gallery(self):
        self.gallery = get_all_gallery()

    def identify(self, probe_emb):
        if not self.gallery:
            return None, 0.0

        best_score = -1.0
        best_name = None

        for entry in self.gallery:
            # Cosine similarity
            score = np.dot(entry["embedding"], probe_emb) / (
                np.linalg.norm(entry["embedding"]) * np.linalg.norm(probe_emb)
            )
            
            if score > best_score:
                best_score = score
                best_name = entry["username"]

        # Adjust threshold: 0.4 - 0.5 is standard for ArcFace
        if best_score > 0.6:
            return best_name, best_score
        return None, best_score