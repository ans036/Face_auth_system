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
    - Adaptive thresholds calculated from gallery statistics
    """
    
    def __init__(
        self,
        face_weight: float = 0.85,
        voice_weight: float = 0.15,
        face_threshold: float = 0.45,            # Raised from 0.30
        voice_threshold: float = 0.15,
        combined_threshold: float = 0.48,        # Raised from 0.30
        face_only_threshold: float = 0.50,       # Raised from 0.32
        confidence_gap: float = 0.05             # Raised from 0.01
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
        
        # Adaptive threshold flag
        self.thresholds_calibrated = False
    
    def load_galleries(self):
        """Load both face and voice galleries and calibrate thresholds."""
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
        
        # Calibrate adaptive thresholds from gallery
        self.calibrate_thresholds()
    
    def calibrate_thresholds(self):
        """Calculate adaptive thresholds from gallery embedding statistics."""
        if len(self.face_gallery) < 2:
            print("⚠️ Need at least 2 users for adaptive calibration, using defaults")
            return
        
        # Group embeddings by user
        user_embeddings = {}
        for entry in self.face_gallery:
            user = entry["username"]
            if user == "Unknown":
                continue  # Skip unknown/negative samples for calibration
            emb = np.frombuffer(entry["embedding"], dtype=np.float32)
            if user not in user_embeddings:
                user_embeddings[user] = []
            user_embeddings[user].append(emb)
        
        if len(user_embeddings) < 2:
            print("⚠️ Need at least 2 real users for adaptive calibration")
            return
        
        # Calculate inter-class similarities (between different users)
        inter_scores = []
        users = list(user_embeddings.keys())
        for i, u1 in enumerate(users):
            for u2 in users[i+1:]:
                for e1 in user_embeddings[u1][:5]:  # Limit to first 5 per user
                    for e2 in user_embeddings[u2][:5]:
                        inter_scores.append(float(np.dot(e1, e2)))
        
        if not inter_scores:
            return
        
        # Calculate intra-class similarities (same user)
        intra_scores = []
        for user, embs in user_embeddings.items():
            if len(embs) >= 2:
                for i, e1 in enumerate(embs[:10]):
                    for e2 in embs[i+1:10]:
                        intra_scores.append(float(np.dot(e1, e2)))
        
        # Calculate statistics
        mean_inter = np.mean(inter_scores)
        std_inter = np.std(inter_scores)
        max_inter = np.max(inter_scores)
        
        mean_intra = np.mean(intra_scores) if intra_scores else 0.7
        
        print(f"📊 Gallery stats: inter-class μ={mean_inter:.3f} σ={std_inter:.3f} max={max_inter:.3f}")
        print(f"📊 Gallery stats: intra-class μ={mean_intra:.3f}")
        
        # Set adaptive thresholds:
        # - Should be above max inter-class similarity
        # - But below mean intra-class similarity
        adaptive_threshold = max(max_inter + 0.03, mean_inter + 2 * std_inter)
        
        # Clamp to reasonable range [0.40, 0.65]
        self.face_only_threshold = max(0.45, min(0.65, adaptive_threshold + 0.05))
        self.face_threshold = max(0.40, min(0.55, adaptive_threshold))
        self.combined_threshold = self.face_threshold + 0.03
        
        # Confidence gap should separate close matches from easy ones
        self.confidence_gap = max(0.03, min(0.10, std_inter))
        
        self.thresholds_calibrated = True
        print(f"🎯 Adaptive thresholds set: face={self.face_threshold:.2f}, face_only={self.face_only_threshold:.2f}, gap={self.confidence_gap:.2f}")
    
    def _softmax_probabilities(self, scores: Dict[str, float], temperature: float = 0.1) -> Dict[str, float]:
        """
        Convert raw similarity scores to softmax probabilities.
        
        This provides a principled way to interpret scores as class probabilities
        that sum to 1, making the decision more robust.
        
        Args:
            scores: Dict of username -> combined_score
            temperature: Controls sharpness of distribution (lower = sharper)
                        0.1 = very confident, 1.0 = more diffuse
        
        Returns:
            Dict of username -> probability (sums to 1.0)
        """
        if not scores:
            return {}
        
        # Extract scores as array
        users = list(scores.keys())
        score_values = np.array([scores[u] for u in users])
        
        # Apply temperature scaling before softmax
        # Lower temperature makes distribution sharper (more confident)
        scaled_scores = score_values / temperature
        
        # Softmax: exp(x) / sum(exp(x))
        # Subtract max for numerical stability
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return {users[i]: float(probabilities[i]) for i in range(len(users))}
    
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
        
        # ===== SOFTMAX PROBABILITY CLASSIFICATION =====
        # Convert combined scores to probabilities using softmax
        score_dict = {user: scores["combined"] for user, scores in combined_scores.items()}
        probabilities = self._softmax_probabilities(score_dict, temperature=0.15)
        
        # Get probability for best user and for Unknown (if exists)
        best_prob = probabilities.get(best_user, 0.0)
        unknown_prob = probabilities.get("Unknown", 0.0)
        
        # Log probabilities for debugging
        print(f"📊 Probabilities: {best_user}={best_prob:.2%} | Unknown={unknown_prob:.2%}")
        
        # Add probabilities to result
        result["probabilities"] = probabilities
        result["best_probability"] = best_prob
        result["unknown_probability"] = unknown_prob
        
        # ===== SOFTMAX-BASED DECISION LOGIC =====
        authenticated = False
        
        # RULE 1: Reject if Unknown wins or has high probability
        if best_user == "Unknown":
            print(f"❌ Best match is 'Unknown' (P={best_prob:.2%}), rejecting")
            result["name"] = "Unknown"
            return result
        
        if unknown_prob > 0.25:
            print(f"⚠️ Unknown probability too high ({unknown_prob:.2%} > 25%), rejecting")
            result["name"] = "Unknown"
            return result
        
        # RULE 2: Require high probability for authentication
        # P(user) > 60% and raw face score > minimum threshold
        min_probability = 0.60  # Must be at least 60% confident
        min_face_score = 0.35   # Minimum raw face similarity
        
        if best_prob >= min_probability and best_scores["face"] >= min_face_score:
            authenticated = True
            print(f"✅ Authenticated via SOFTMAX: P({best_user})={best_prob:.2%} >= {min_probability:.0%}")
        
        # RULE 3: Very strong face can override (for cases with few users)
        elif best_scores["face"] >= self.face_only_threshold and best_prob >= 0.45:
            authenticated = True
            print(f"✅ Authenticated via STRONG FACE: score={best_scores['face']:.3f}, P={best_prob:.2%}")
        
        # RULE 4: Voice boost for borderline cases
        elif best_scores["face"] >= self.face_threshold and best_scores["voice"] is not None:
            if best_scores["voice"] >= self.voice_threshold and best_prob >= 0.50:
                authenticated = True
                print(f"✅ Authenticated via VOICE BOOST: P={best_prob:.2%}")
        
        if authenticated:
            result["name"] = best_user
            result["is_authenticated"] = True
        else:
            print(f"❌ Not authenticated. P({best_user})={best_prob:.2%} (need ≥60%), Face={best_scores['face']:.3f}")
        
        return result


# Singleton
_multimodal_auth = None

def get_multimodal_authenticator() -> MultimodalAuthenticator:
    """Get or create multimodal authenticator singleton."""
    global _multimodal_auth
    if _multimodal_auth is None:
        _multimodal_auth = MultimodalAuthenticator()
    return _multimodal_auth
