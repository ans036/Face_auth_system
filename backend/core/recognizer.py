import numpy as np
from db.crud import get_all_gallery

class Recognizer:
    def __init__(self, threshold=0.50, confidence_gap=0.03):
        """
        Enhanced recognizer with confidence gap requirement.
        
        Args:
            threshold: Minimum similarity score to accept (0.50)
            confidence_gap: Minimum difference between top 2 candidates (3%)
        """
        self.threshold = threshold
        self.confidence_gap = confidence_gap
        self.gallery = []

    def load_gallery(self):
        """Loads gallery from database."""
        try:
            self.gallery = get_all_gallery()
            if self.gallery:
                print(f"✅ Loaded {len(self.gallery)} embeddings into recognizer")
        except Exception as e:
            print(f"⚠️  Gallery not ready yet: {e}")
            self.gallery = []

    def identify(self, probe_emb):
        """
        Identify a face with enhanced confidence scoring.
        
        Returns:
            (name, confidence) tuple
        """
        # Lazy loading
        if not self.gallery:
            self.load_gallery()
        
        if not self.gallery:
            return "Unknown", 0.0

        # 1. Collect all scores per user (embeddings already normalized by embedder)
        user_scores = {}

        for entry in self.gallery:
            target_emb = np.frombuffer(entry["embedding"], dtype=np.float32)
            
            # Cosine similarity (both embeddings already L2-normalized)
            score = float(np.dot(target_emb, probe_emb))
            
            username = entry["username"]
            if username not in user_scores:
                user_scores[username] = []
            user_scores[username].append(score)

        if not user_scores:
            return "Unknown", 0.0

        # 2. Aggregate scores per user (Top-3 average with fewer embeddings now)
        user_final_scores = {}
        for username, scores in user_scores.items():
            scores.sort(reverse=True)
            # Take average of top 3 scores (or all if less than 3)
            top_k = min(3, len(scores))
            user_final_scores[username] = float(np.mean(scores[:top_k]))

        # 3. Get top 2 candidates
        sorted_candidates = sorted(user_final_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_name = sorted_candidates[0][0]
        best_score = sorted_candidates[0][1]
        
        # Get second best (if exists)
        second_best_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0
        
        # 4. Apply confidence gap requirement
        gap = best_score - second_best_score
        
        # Debug logging (you can comment this out in production)
        if len(sorted_candidates) >= 2:
            print(f"🔍 Top-2: {best_name}={best_score:.3f}, {sorted_candidates[1][0]}={second_best_score:.3f}, Gap={gap:.3f}")
        
        # 5. Decision logic with confidence gap
        if best_score >= self.threshold:
            # Check if winner is clearly ahead of runner-up
            if gap >= self.confidence_gap:
                return best_name, best_score
            else:
                # Too close to call - reject as Unknown
                print(f"⚠️  Rejected: Gap too small ({gap:.3f} < {self.confidence_gap})")
                return "Unknown", best_score
        
        return "Unknown", best_score
