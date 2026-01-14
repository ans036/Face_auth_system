import numpy as np
from db.crud import get_all_gallery

class Recognizer:
    def __init__(self, threshold=0.60): # Increase threshold to 0.60 for better security
        self.threshold = threshold
        self.gallery = []
        self.load_gallery()

    def load_gallery(self):
        self.gallery = get_all_gallery()

    def identify(self, probe_emb):
        if not self.gallery:
            return "Unknown", 0.0

        best_score = -1.0
        best_name = "Unknown"

        # 1. Normalize the incoming probe vector (the face in front of the camera)
        probe_emb = probe_emb / np.linalg.norm(probe_emb)

        for entry in self.gallery:
            # 2. Extract and normalize the stored gallery vector
            target_emb = np.frombuffer(entry["embedding"], dtype=np.float32)
            target_emb = target_emb / np.linalg.norm(target_emb)
            
            # 3. Calculate Cosine Similarity (Result will be between -1.0 and 1.0)
            score = np.dot(target_emb, probe_emb)
            
            if score > best_score:
                best_score = score
                best_name = entry["username"]

        # 4. Apply strict thresholding to distinguish you from your friend
        if best_score >= self.threshold:
            return best_name, float(best_score)
        
        return "Unknown", float(best_score)