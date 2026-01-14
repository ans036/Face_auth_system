import numpy as np
from db.crud import get_all_gallery

class Recognizer:
    def __init__(self, threshold=0.60):
        self.threshold = threshold
        self.gallery = []
        self.load_gallery()

    def load_gallery(self):
        self.gallery = get_all_gallery()

    def identify(self, probe_emb):
        if not self.gallery:
            return "Unknown", 0.0

        # 1. Normalize the incoming probe vector
        probe_emb = probe_emb / np.linalg.norm(probe_emb)
        
        user_scores = {} # Store sums of scores per user
        user_counts = {} # Store how many templates per user

        for entry in self.gallery:
            target_emb = np.frombuffer(entry["embedding"], dtype=np.float32)
            # Ensure gallery embedding is normalized
            target_emb = target_emb / np.linalg.norm(target_emb)
            
            # 2. Calculate Cosine Similarity
            score = np.dot(target_emb, probe_emb)
            
            username = entry["username"]
            if username not in user_scores:
                user_scores[username] = []
            user_scores[username].append(score)

        best_final_score = -1.0
        identified_name = "Unknown"

        # 3. Weighted Scoring: Look at the top 3 best images for EACH person
        for username, scores in user_scores.items():
            scores.sort(reverse=True)
            # Average of top 3 matches for this specific person
            top_3_avg = np.mean(scores[:3]) if len(scores) >= 3 else scores[0]
            
            if top_3_avg > best_final_score:
                best_final_score = top_3_avg
                identified_name = username

        # 4. Final strict threshold check
        if best_final_score >= self.threshold:
            return identified_name, float(best_final_score)
        
        return "Unknown", float(best_final_score)