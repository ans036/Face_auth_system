"""
Multimodal Fusion Module
Combines face and voice biometrics for robust authentication.

Fusion Strategy: Score-level fusion with weighted average
- Face: 60% weight (primary biometric)
- Voice: 40% weight (secondary, optional)

Fallback: Face-only if voice unavailable
"""

import numpy as np
from typing import Dict, Optional, Tuple
from db.crud import get_all_gallery, get_voice_gallery


class MultimodalAuthenticator:
    """
    Combines face and voice recognition for robust authentication.
    
    Adaptive fusion design:
    - Face is PRIMARY (voice only helps, never hurts)
    - Strong face match authenticates alone
    - Voice can boost confidence but won't block
    """
    
    def __init__(
        self,
        face_weight: float = 0.85,              # Increased from 0.75
        voice_weight: float = 0.15,              # Decreased from 0.25
        face_threshold: float = 0.30,            # Lowered from 0.35
        voice_threshold: float = 0.10,           # Lowered from 0.25
        combined_threshold: float = 0.30,        # Lowered from 0.40
        face_only_threshold: float = 0.32,       # Lowered from 0.50
        confidence_gap: float = 0.01             # Lowered from 0.02
    ):
        self.face_weight = face_weight
        self.voice_weight = voice_weight
        self.face_threshold = face_threshold
        self.voice_threshold = voice_threshold
        self.combined_threshold = combined_threshold
        self.face_only_threshold = face_only_threshold
        self.confidence_gap = confidence_gap
        
        # Cached galleries
        self.face_gallery = []
        self.voice_gallery = []
    
    def load_galleries(self):
        """Load both face and voice galleries."""
        try:
            self.face_gallery = get_all_gallery()
            print(f"✅ Loaded {len(self.face_gallery)} face embeddings")
        except Exception as e:
            print(f"⚠️ Failed to load face gallery: {e}")
            self.face_gallery = []
        
        try:
            self.voice_gallery = get_voice_gallery()
            print(f"✅ Loaded {len(self.voice_gallery)} voice embeddings")
        except Exception as e:
            print(f"⚠️ Voice gallery not available: {e}")
            self.voice_gallery = []
    
    def _get_face_scores(self, probe_emb: np.ndarray) -> Dict[str, float]:
        """Get per-user face similarity scores."""
        user_scores = {}
        
        for entry in self.face_gallery:
            target_emb = np.frombuffer(entry["embedding"], dtype=np.float32)
            score = float(np.dot(target_emb, probe_emb))
            
            username = entry["username"]
            if username not in user_scores:
                user_scores[username] = []
            user_scores[username].append(score)
        
        # Average top-5 scores per user
        final_scores = {}
        for username, scores in user_scores.items():
            scores.sort(reverse=True)
            top_k = min(5, len(scores))
            final_scores[username] = float(np.mean(scores[:top_k]))
        
        return final_scores
    
    def _get_voice_scores(self, probe_emb: np.ndarray) -> Dict[str, float]:
        """Get per-user voice similarity scores."""
        user_scores = {}
        
        for entry in self.voice_gallery:
            target_emb = np.frombuffer(entry["embedding"], dtype=np.float32)
            score = float(np.dot(target_emb, probe_emb))
            
            username = entry["username"]
            if username not in user_scores:
                user_scores[username] = []
            user_scores[username].append(score)
        
        # Average all scores per user
        final_scores = {}
        for username, scores in user_scores.items():
            final_scores[username] = float(np.mean(scores))
        
        return final_scores
    
    def identify(
        self,
        face_emb: Optional[np.ndarray] = None,
        voice_emb: Optional[np.ndarray] = None,
        liveness_passed: bool = False
    ) -> Dict:
        """
        Identify user using adaptive multimodal biometrics.
        
        Decision Logic (Adaptive):
        1. Strong face alone (>0.50) → Authenticate
        2. Face + Voice combined (>0.40) with face > 0.30 → Authenticate
        3. Voice is supplementary, not a blocker
        """
        result = {
            "name": "Unknown",
            "face_score": 0.0,
            "voice_score": None,
            "combined_score": 0.0,
            "is_authenticated": False,
            "modalities_used": [],
            "liveness_passed": liveness_passed
        }
        
        # Require face at minimum
        if face_emb is None:
            return result
        
        # Lazy load galleries
        if not self.face_gallery:
            self.load_galleries()
        
        if not self.face_gallery:
            return result
        
        # Get face scores
        face_scores = self._get_face_scores(face_emb)
        result["modalities_used"].append("face")
        
        # Get voice scores if available
        voice_scores = {}
        if voice_emb is not None and self.voice_gallery:
            voice_scores = self._get_voice_scores(voice_emb)
            result["modalities_used"].append("voice")
        
        # Fuse scores
        combined_scores = {}
        all_users = set(face_scores.keys()) | set(voice_scores.keys())
        
        for user in all_users:
            face_score = face_scores.get(user, 0.0)
            voice_score = voice_scores.get(user, None)
            
            # ADAPTIVE FUSION: Voice only helps, never hurts
            if voice_score is not None and voice_score > 0.1:
                # Only include voice if it's actually helpful (>10% match)
                combined = (self.face_weight * face_score + 
                           self.voice_weight * voice_score)
            else:
                # Face only (voice is absent or unhelpful)
                combined = face_score
            
            combined_scores[user] = {
                "face": face_score,
                "voice": voice_score,
                "combined": combined
            }
        
        if not combined_scores:
            return result
        
        # Find best match
        sorted_users = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["combined"],
            reverse=True
        )
        
        best_user = sorted_users[0][0]
        best_scores = sorted_users[0][1]
        
        result["face_score"] = best_scores["face"]
        result["voice_score"] = best_scores["voice"]
        result["combined_score"] = best_scores["combined"]
        
        # Debug logging
        print(f"🔍 Best match: {best_user} | Face: {best_scores['face']:.3f} | Voice: {best_scores['voice'] if best_scores['voice'] else 'N/A'} | Combined: {best_scores['combined']:.3f}")
        
        # Confidence gap check
        gap = 0.0
        if len(sorted_users) > 1:
            gap = best_scores["combined"] - sorted_users[1][1]["combined"]
        gap_passed = gap >= self.confidence_gap or len(sorted_users) == 1
        
        # ADAPTIVE DECISION LOGIC
        authenticated = False
        
        # Rule 1: Strong face alone can authenticate
        if best_scores["face"] >= self.face_only_threshold:
            authenticated = True
            print(f"✅ Authenticated via STRONG FACE (>{self.face_only_threshold})")
        
        # Rule 2: Combined score with decent face
        elif best_scores["combined"] >= self.combined_threshold and best_scores["face"] >= self.face_threshold:
            authenticated = True
            print(f"✅ Authenticated via COMBINED SCORE (>{self.combined_threshold})")
        
        # Rule 3: Voice boosts borderline face
        elif best_scores["face"] >= self.face_threshold and best_scores["voice"] is not None and best_scores["voice"] >= self.voice_threshold:
            authenticated = True
            print(f"✅ Authenticated via FACE + VOICE boost")
        
        # Apply gap check
        if authenticated and not gap_passed:
            print(f"⚠️ Gap too small ({gap:.3f} < {self.confidence_gap}), rejecting")
            authenticated = False
        
        if authenticated:
            result["name"] = best_user
            result["is_authenticated"] = True
        else:
            print(f"❌ Not authenticated. Face: {best_scores['face']:.3f} (threshold: {self.face_threshold}), Combined: {best_scores['combined']:.3f} (threshold: {self.combined_threshold})")
        
        return result


# Singleton
_multimodal_auth = None

def get_multimodal_authenticator() -> MultimodalAuthenticator:
    """Get or create multimodal authenticator singleton."""
    global _multimodal_auth
    if _multimodal_auth is None:
        _multimodal_auth = MultimodalAuthenticator()
    return _multimodal_auth
