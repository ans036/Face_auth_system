"""
Biometric Evaluation Script
Calculates FAR, FRR, EER, and generates ROC curves following NIST guidelines.

This script provides comprehensive biometric accuracy testing for:
- Face recognition accuracy
- Voice verification accuracy
- Multimodal fusion performance

Output:
- ROC curves (publication-ready)
- DET curves
- FAR/FRR tables
- EER calculation

Usage:
    python scripts/biometric_evaluation.py
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json

# Ensure output directory exists
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    eer: float
    eer_threshold: float
    far_at_frr_01: float  # FAR when FRR = 1%
    far_at_frr_1: float   # FAR when FRR = 10%
    auc: float            # Area Under ROC Curve
    thresholds: List[float]
    far_values: List[float]
    frr_values: List[float]


class BiometricEvaluator:
    """
    Evaluates biometric system accuracy using genuine/impostor score methodology.
    
    Following NIST evaluation protocols for:
    - False Accept Rate (FAR): P(accept | impostor)
    - False Reject Rate (FRR): P(reject | genuine)
    - Equal Error Rate (EER): Point where FAR = FRR
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.genuine_scores: List[float] = []
        self.impostor_scores: List[float] = []
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def generate_score_distributions(
        self, 
        embeddings: Dict[str, List[np.ndarray]],
        max_pairs: int = 10000
    ) -> Tuple[List[float], List[float]]:
        """
        Generate genuine and impostor score distributions.
        
        Args:
            embeddings: Dict mapping username to list of embeddings
            max_pairs: Maximum pairs to sample (for large datasets)
            
        Returns:
            (genuine_scores, impostor_scores)
        """
        genuine_scores = []
        impostor_scores = []
        
        users = list(embeddings.keys())
        
        # Genuine comparisons (same user, different samples)
        for user in users:
            user_embeddings = embeddings[user]
            if len(user_embeddings) < 2:
                continue
            
            # Compare all pairs within user
            for i in range(len(user_embeddings)):
                for j in range(i + 1, len(user_embeddings)):
                    score = self.cosine_similarity(user_embeddings[i], user_embeddings[j])
                    genuine_scores.append(score)
        
        # Impostor comparisons (different users)
        impostor_count = 0
        for i, user1 in enumerate(users):
            for j, user2 in enumerate(users):
                if i >= j:
                    continue
                
                # Compare first embedding of each user (or sample if many)
                for emb1 in embeddings[user1][:3]:  # Limit to 3 samples per pair
                    for emb2 in embeddings[user2][:3]:
                        score = self.cosine_similarity(emb1, emb2)
                        impostor_scores.append(score)
                        impostor_count += 1
                        
                        if impostor_count >= max_pairs:
                            break
                    if impostor_count >= max_pairs:
                        break
                if impostor_count >= max_pairs:
                    break
            if impostor_count >= max_pairs:
                break
        
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        
        return genuine_scores, impostor_scores
    
    def calculate_far_frr(
        self, 
        genuine: List[float], 
        impostor: List[float], 
        threshold: float
    ) -> Tuple[float, float]:
        """
        Calculate FAR and FRR at a given threshold.
        
        Args:
            genuine: Genuine (same-person) scores
            impostor: Impostor (different-person) scores
            threshold: Decision threshold
            
        Returns:
            (FAR, FRR)
        """
        # FAR: proportion of impostors accepted (score >= threshold)
        far = sum(1 for s in impostor if s >= threshold) / max(len(impostor), 1)
        
        # FRR: proportion of genuine rejected (score < threshold)
        frr = sum(1 for s in genuine if s < threshold) / max(len(genuine), 1)
        
        return far, frr
    
    def find_eer(
        self, 
        genuine: List[float], 
        impostor: List[float],
        num_thresholds: int = 1000
    ) -> Tuple[float, float, List[float], List[float], List[float]]:
        """
        Find Equal Error Rate (EER) where FAR = FRR.
        
        Args:
            genuine: Genuine scores
            impostor: Impostor scores
            num_thresholds: Number of threshold points to evaluate
            
        Returns:
            (eer, eer_threshold, thresholds, far_values, frr_values)
        """
        # Generate threshold range
        all_scores = genuine + impostor
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 1
        
        thresholds = np.linspace(min_score, max_score, num_thresholds)
        
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            far, frr = self.calculate_far_frr(genuine, impostor, threshold)
            far_values.append(far)
            frr_values.append(frr)
        
        # Find EER (where FAR ≈ FRR)
        min_diff = float('inf')
        eer = 0.0
        eer_threshold = 0.5
        
        for i, (far, frr) in enumerate(zip(far_values, frr_values)):
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
                eer_threshold = thresholds[i]
        
        return eer, eer_threshold, list(thresholds), far_values, frr_values
    
    def calculate_auc(self, far_values: List[float], frr_values: List[float]) -> float:
        """Calculate Area Under ROC Curve using trapezoidal rule."""
        # ROC: TPR (1-FRR) vs FPR (FAR)
        tpr = [1 - frr for frr in frr_values]
        fpr = far_values
        
        # Sort by FPR for proper integration
        sorted_pairs = sorted(zip(fpr, tpr))
        fpr_sorted = [p[0] for p in sorted_pairs]
        tpr_sorted = [p[1] for p in sorted_pairs]
        
        # Trapezoidal integration
        auc = 0.0
        for i in range(1, len(fpr_sorted)):
            auc += (fpr_sorted[i] - fpr_sorted[i-1]) * (tpr_sorted[i] + tpr_sorted[i-1]) / 2
        
        return auc
    
    def evaluate(
        self, 
        embeddings: Dict[str, List[np.ndarray]]
    ) -> EvaluationResult:
        """
        Run full evaluation on embedding dataset.
        
        Args:
            embeddings: Dict mapping username to list of embeddings
            
        Returns:
            EvaluationResult with all metrics
        """
        # Generate score distributions
        genuine, impostor = self.generate_score_distributions(embeddings)
        
        if not genuine or not impostor:
            raise ValueError("Insufficient data for evaluation")
        
        # Calculate EER
        eer, eer_threshold, thresholds, far_values, frr_values = self.find_eer(genuine, impostor)
        
        # Calculate AUC
        auc = self.calculate_auc(far_values, frr_values)
        
        # Find FAR at specific FRR operating points
        # FAR @ FRR = 1%
        far_at_frr_01 = 1.0
        for i, frr in enumerate(frr_values):
            if frr <= 0.01:
                far_at_frr_01 = far_values[i]
                break
        
        # FAR @ FRR = 10%
        far_at_frr_1 = 1.0
        for i, frr in enumerate(frr_values):
            if frr <= 0.10:
                far_at_frr_1 = far_values[i]
                break
        
        return EvaluationResult(
            eer=eer,
            eer_threshold=eer_threshold,
            far_at_frr_01=far_at_frr_01,
            far_at_frr_1=far_at_frr_1,
            auc=auc,
            thresholds=thresholds,
            far_values=far_values,
            frr_values=frr_values
        )


def generate_synthetic_embeddings(
    num_users: int = 20,
    samples_per_user: int = 5,
    embedding_dim: int = 512,
    intra_class_variance: float = 0.1,
    inter_class_distance: float = 1.0
) -> Dict[str, List[np.ndarray]]:
    """
    Generate synthetic embeddings for testing.
    
    Each user has a unique "centroid" embedding, with variations simulating
    different captures of the same person.
    
    Args:
        num_users: Number of unique identities
        samples_per_user: Samples per user
        embedding_dim: Dimension of embeddings
        intra_class_variance: Variance within same user
        inter_class_distance: Distance between user centroids
        
    Returns:
        Dict mapping username to list of embeddings
    """
    embeddings = {}
    
    for i in range(num_users):
        # Generate user centroid
        centroid = np.random.randn(embedding_dim).astype(np.float32)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)  # Normalize
        
        # Generate variations
        user_embeddings = []
        for j in range(samples_per_user):
            # Add noise
            noise = np.random.randn(embedding_dim).astype(np.float32) * intra_class_variance
            sample = centroid + noise
            sample = sample / (np.linalg.norm(sample) + 1e-8)  # Re-normalize
            user_embeddings.append(sample)
        
        embeddings[f"user_{i:03d}"] = user_embeddings
    
    return embeddings


def plot_roc_curve(result: EvaluationResult, output_path: Path, title: str = "ROC Curve"):
    """Generate publication-quality ROC curve."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # TPR = 1 - FRR, FPR = FAR
        tpr = [1 - frr for frr in result.frr_values]
        fpr = result.far_values
        
        # Plot ROC
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {result.auc:.4f})')
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
        
        # Mark EER point
        eer_tpr = 1 - result.eer
        ax.plot(result.eer, eer_tpr, 'ro', markersize=10, label=f'EER = {result.eer:.2%}')
        
        ax.set_xlabel('False Positive Rate (FAR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (1-FRR)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_path}")
        
    except ImportError:
        print("⚠️ matplotlib/seaborn not available, skipping ROC plot")


def plot_det_curve(result: EvaluationResult, output_path: Path, title: str = "DET Curve"):
    """Generate Detection Error Trade-off (DET) curve."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import norm
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Convert to probit scale (DET uses normal deviate scale)
        # Filter out 0 and 1 values to avoid infinity
        far_filtered = []
        frr_filtered = []
        for far, frr in zip(result.far_values, result.frr_values):
            if 0 < far < 1 and 0 < frr < 1:
                far_filtered.append(far)
                frr_filtered.append(frr)
        
        if far_filtered and frr_filtered:
            # Convert to probit scale
            far_probit = norm.ppf(far_filtered)
            frr_probit = norm.ppf(frr_filtered)
            
            ax.plot(far_probit, frr_probit, 'b-', linewidth=2, label='DET Curve')
            
            # Custom tick labels
            tick_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
            tick_labels = ['0.1%', '1%', '5%', '10%', '20%', '40%']
            tick_positions = [norm.ppf(v) for v in tick_values]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            
            ax.set_xlabel('False Accept Rate (FAR)', fontsize=12)
            ax.set_ylabel('False Reject Rate (FRR)', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved: {output_path}")
        else:
            print("⚠️ Insufficient data for DET curve")
        
    except ImportError:
        print("⚠️ matplotlib/seaborn/scipy not available, skipping DET plot")


def plot_score_distributions(
    genuine: List[float], 
    impostor: List[float], 
    eer_threshold: float,
    output_path: Path
):
    """Plot genuine and impostor score distributions."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distributions
        sns.histplot(genuine, bins=50, alpha=0.6, label='Genuine', color='green', kde=True, ax=ax)
        sns.histplot(impostor, bins=50, alpha=0.6, label='Impostor', color='red', kde=True, ax=ax)
        
        # Mark EER threshold
        ax.axvline(eer_threshold, color='blue', linestyle='--', linewidth=2, 
                   label=f'EER Threshold = {eer_threshold:.3f}')
        
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Genuine vs Impostor Score Distributions', fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_path}")
        
    except ImportError:
        print("⚠️ matplotlib/seaborn not available, skipping distribution plot")


def generate_metrics_report(
    face_result: EvaluationResult,
    voice_result: Optional[EvaluationResult],
    output_path: Path
):
    """Generate Markdown report with all metrics."""
    report = """# Biometric Evaluation Report

## Summary

| Metric | Face Recognition | Voice Verification |
|--------|------------------|-------------------|
"""
    
    face_str = f"{face_result.eer:.2%}"
    voice_str = f"{voice_result.eer:.2%}" if voice_result else "N/A"
    report += f"| **EER** | {face_str} | {voice_str} |\n"
    
    face_str = f"{face_result.eer_threshold:.3f}"
    voice_str = f"{voice_result.eer_threshold:.3f}" if voice_result else "N/A"
    report += f"| EER Threshold | {face_str} | {voice_str} |\n"
    
    face_str = f"{face_result.auc:.4f}"
    voice_str = f"{voice_result.auc:.4f}" if voice_result else "N/A"
    report += f"| AUC | {face_str} | {voice_str} |\n"
    
    face_str = f"{face_result.far_at_frr_01:.2%}"
    voice_str = f"{voice_result.far_at_frr_01:.2%}" if voice_result else "N/A"
    report += f"| FAR @ 1% FRR | {face_str} | {voice_str} |\n"
    
    face_str = f"{face_result.far_at_frr_1:.2%}"
    voice_str = f"{voice_result.far_at_frr_1:.2%}" if voice_result else "N/A"
    report += f"| FAR @ 10% FRR | {face_str} | {voice_str} |\n"
    
    report += """
## Interpretation

- **EER (Equal Error Rate)**: The operating point where FAR = FRR. Lower is better.
- **AUC (Area Under Curve)**: Closer to 1.0 indicates better discrimination. 
- **FAR @ 1% FRR**: False accept rate when false reject rate is 1% (high security setting).
- **FAR @ 10% FRR**: False accept rate when false reject rate is 10% (convenience setting).

## Generated Plots

1. `face_roc_curve.png` - ROC curve for face recognition
2. `face_det_curve.png` - DET curve for face recognition  
3. `face_score_distributions.png` - Genuine vs impostor score distributions
4. `voice_roc_curve.png` - ROC curve for voice verification (if available)

## NIST Compliance Notes

This evaluation follows NIST FRVT (Face Recognition Vendor Test) methodology:
- Genuine comparisons: All same-subject pairs
- Impostor comparisons: Cross-subject pairs (sampled)
- Metrics calculated at multiple operating points
"""
    
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"📝 Saved: {output_path}")


def main():
    print("=" * 60)
    print("Biometric Evaluation - FAR/FRR Analysis")
    print("=" * 60)
    
    # Generate synthetic data for demonstration
    # In production, load real embeddings from database
    print("\n📊 Generating synthetic embeddings for evaluation...")
    
    # Face embeddings (512-dim)
    face_embeddings = generate_synthetic_embeddings(
        num_users=50,
        samples_per_user=10,
        embedding_dim=512,
        intra_class_variance=0.15,  # Realistic intra-class variance
        inter_class_distance=1.0
    )
    print(f"   Generated {len(face_embeddings)} users with face embeddings")
    
    # Voice embeddings (192-dim)
    voice_embeddings = generate_synthetic_embeddings(
        num_users=50,
        samples_per_user=5,
        embedding_dim=192,
        intra_class_variance=0.20,  # Voice has higher variance
        inter_class_distance=1.0
    )
    print(f"   Generated {len(voice_embeddings)} users with voice embeddings")
    
    # Evaluate face recognition
    print("\n🔍 Evaluating face recognition accuracy...")
    face_evaluator = BiometricEvaluator(embedding_dim=512)
    face_result = face_evaluator.evaluate(face_embeddings)
    
    print(f"   EER: {face_result.eer:.2%}")
    print(f"   EER Threshold: {face_result.eer_threshold:.3f}")
    print(f"   AUC: {face_result.auc:.4f}")
    
    # Evaluate voice verification
    print("\n🎤 Evaluating voice verification accuracy...")
    voice_evaluator = BiometricEvaluator(embedding_dim=192)
    voice_result = voice_evaluator.evaluate(voice_embeddings)
    
    print(f"   EER: {voice_result.eer:.2%}")
    print(f"   EER Threshold: {voice_result.eer_threshold:.3f}")
    print(f"   AUC: {voice_result.auc:.4f}")
    
    # Generate plots
    print("\n📊 Generating plots...")
    
    plot_roc_curve(face_result, OUTPUT_DIR / "face_roc_curve.png", "Face Recognition ROC Curve")
    plot_det_curve(face_result, OUTPUT_DIR / "face_det_curve.png", "Face Recognition DET Curve")
    plot_score_distributions(
        face_evaluator.genuine_scores, 
        face_evaluator.impostor_scores,
        face_result.eer_threshold,
        OUTPUT_DIR / "face_score_distributions.png"
    )
    
    plot_roc_curve(voice_result, OUTPUT_DIR / "voice_roc_curve.png", "Voice Verification ROC Curve")
    plot_score_distributions(
        voice_evaluator.genuine_scores,
        voice_evaluator.impostor_scores,
        voice_result.eer_threshold,
        OUTPUT_DIR / "voice_score_distributions.png"
    )
    
    # Generate report
    print("\n📝 Generating metrics report...")
    generate_metrics_report(face_result, voice_result, OUTPUT_DIR / "biometric_evaluation_report.md")
    
    # Save raw results as JSON
    results_json = {
        "face": {
            "eer": face_result.eer,
            "eer_threshold": face_result.eer_threshold,
            "auc": face_result.auc,
            "far_at_frr_01": face_result.far_at_frr_01,
            "far_at_frr_1": face_result.far_at_frr_1,
            "num_genuine_pairs": len(face_evaluator.genuine_scores),
            "num_impostor_pairs": len(face_evaluator.impostor_scores),
        },
        "voice": {
            "eer": voice_result.eer,
            "eer_threshold": voice_result.eer_threshold,
            "auc": voice_result.auc,
            "far_at_frr_01": voice_result.far_at_frr_01,
            "far_at_frr_1": voice_result.far_at_frr_1,
            "num_genuine_pairs": len(voice_evaluator.genuine_scores),
            "num_impostor_pairs": len(voice_evaluator.impostor_scores),
        }
    }
    
    with open(OUTPUT_DIR / "biometric_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"📊 Saved: {OUTPUT_DIR / 'biometric_results.json'}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete! Results in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
