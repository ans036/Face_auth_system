"""
Tests for biometric evaluation metrics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestBiometricEvaluator:
    """Test cases for BiometricEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        from biometric_evaluation import BiometricEvaluator
        return BiometricEvaluator(embedding_dim=512)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        embeddings = {}
        np.random.seed(42)  # Reproducible
        
        for i in range(5):
            centroid = np.random.randn(512).astype(np.float32)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            
            samples = []
            for j in range(3):
                noise = np.random.randn(512).astype(np.float32) * 0.1
                sample = centroid + noise
                sample = sample / (np.linalg.norm(sample) + 1e-8)
                samples.append(sample)
            
            embeddings[f"user_{i}"] = samples
        
        return embeddings
    
    def test_cosine_similarity_identical(self, evaluator):
        """Identical embeddings should have similarity of 1.0."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        similarity = evaluator.cosine_similarity(emb, emb)
        
        assert abs(similarity - 1.0) < 1e-5
    
    def test_cosine_similarity_orthogonal(self, evaluator):
        """Orthogonal embeddings should have similarity of 0."""
        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0
        
        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0
        
        similarity = evaluator.cosine_similarity(emb1, emb2)
        
        assert abs(similarity) < 1e-5
    
    def test_cosine_similarity_opposite(self, evaluator):
        """Opposite embeddings should have similarity of -1.0."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        
        emb2 = -emb1
        
        similarity = evaluator.cosine_similarity(emb1, emb2)
        
        assert abs(similarity + 1.0) < 1e-5
    
    def test_generate_score_distributions(self, evaluator, sample_embeddings):
        """Should generate genuine and impostor score distributions."""
        genuine, impostor = evaluator.generate_score_distributions(sample_embeddings)
        
        # Should have scores
        assert len(genuine) > 0
        assert len(impostor) > 0
        
        # Genuine should generally be higher than impostor
        avg_genuine = np.mean(genuine)
        avg_impostor = np.mean(impostor)
        assert avg_genuine > avg_impostor
    
    def test_calculate_far_frr_zero_threshold(self, evaluator):
        """With threshold 0, should accept everything (FAR=1, FRR=0)."""
        genuine = [0.8, 0.9, 0.7]
        impostor = [0.3, 0.4, 0.2]
        
        far, frr = evaluator.calculate_far_frr(genuine, impostor, threshold=0)
        
        assert far == 1.0  # All impostors accepted
        assert frr == 0.0  # No genuine rejected
    
    def test_calculate_far_frr_high_threshold(self, evaluator):
        """With threshold 1, should reject everything (FAR=0, FRR=1)."""
        genuine = [0.8, 0.9, 0.7]
        impostor = [0.3, 0.4, 0.2]
        
        far, frr = evaluator.calculate_far_frr(genuine, impostor, threshold=1)
        
        assert far == 0.0  # No impostors accepted
        assert frr == 1.0  # All genuine rejected
    
    def test_find_eer(self, evaluator, sample_embeddings):
        """Should find EER between 0 and 1."""
        genuine, impostor = evaluator.generate_score_distributions(sample_embeddings)
        
        eer, eer_threshold, thresholds, far_values, frr_values = evaluator.find_eer(genuine, impostor)
        
        assert 0 <= eer <= 1
        assert len(thresholds) > 0
        assert len(far_values) == len(thresholds)
        assert len(frr_values) == len(thresholds)
    
    def test_evaluate_full(self, evaluator, sample_embeddings):
        """Full evaluation should return valid result."""
        result = evaluator.evaluate(sample_embeddings)
        
        assert 0 <= result.eer <= 1
        assert 0 <= result.auc <= 1
        assert len(result.thresholds) > 0
        assert result.far_at_frr_01 >= 0
        assert result.far_at_frr_1 >= 0


class TestEvaluationResult:
    """Test EvaluationResult data structure."""
    
    def test_evaluation_result_creation(self):
        """Should create EvaluationResult with all fields."""
        from biometric_evaluation import EvaluationResult
        
        result = EvaluationResult(
            eer=0.05,
            eer_threshold=0.5,
            far_at_frr_01=0.01,
            far_at_frr_1=0.1,
            auc=0.95,
            thresholds=[0.1, 0.5, 0.9],
            far_values=[0.9, 0.5, 0.1],
            frr_values=[0.1, 0.5, 0.9]
        )
        
        assert result.eer == 0.05
        assert result.auc == 0.95


class TestSyntheticEmbeddings:
    """Test synthetic embedding generation."""
    
    def test_generate_synthetic_embeddings(self):
        """Should generate embeddings for specified users."""
        from biometric_evaluation import generate_synthetic_embeddings
        
        embeddings = generate_synthetic_embeddings(
            num_users=10,
            samples_per_user=3,
            embedding_dim=512
        )
        
        assert len(embeddings) == 10
        for user, samples in embeddings.items():
            assert len(samples) == 3
            for sample in samples:
                assert sample.shape == (512,)
                # Should be normalized
                assert abs(np.linalg.norm(sample) - 1.0) < 1e-5
