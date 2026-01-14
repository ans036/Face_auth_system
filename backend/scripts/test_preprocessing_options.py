#!/usr/bin/env python3
"""
Test different preprocessing options to find what works best
"""

import sys
sys.path.insert(0, '/app')

import cv2
import numpy as np
from core.embedder import FaceEmbedder
from core.detector import FaceDetector
from utils.image import crop_box
from utils.allignment import align_face

def apply_clahe(img, clip_limit=3.0):
    """Apply CLAHE preprocessing with configurable clip limit"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    img_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

def test_with_settings(clahe_enabled, clip_limit=3.0):
    """Test embedding generation with specific settings"""
    
    detector = FaceDetector()
    embedder = FaceEmbedder()
    
    # Test with Anish
    img_anish = cv2.imread("/app/database/Anish/pic1.jpeg")
    if clahe_enabled:
        img_anish = apply_clahe(img_anish, clip_limit)
    img_anish_rgb = cv2.cvtColor(img_anish, cv2.COLOR_BGR2RGB)
    box_anish, kps_anish = detector.detect(img_anish_rgb)
    aligned_anish = align_face(img_anish_rgb, kps_anish)
    face_anish = crop_box(aligned_anish, box_anish)
    emb_anish = embedder.embed(face_anish)  # Already normalized
    
    # Test with Sayani
    img_sayani = cv2.imread("/app/database/Sayani/pic1.jpeg")
    if clahe_enabled:
        img_sayani = apply_clahe(img_sayani, clip_limit)
    img_sayani_rgb = cv2.cvtColor(img_sayani, cv2.COLOR_BGR2RGB)
    box_sayani, kps_sayani = detector.detect(img_sayani_rgb)
    aligned_sayani = align_face(img_sayani_rgb, kps_sayani)
    face_sayani = crop_box(aligned_sayani, box_sayani)
    emb_sayani = embedder.embed(face_sayani)  # Already normalized
    
    # Compare
    similarity = float(np.dot(emb_anish, emb_sayani))
    
    # Calculate embedding statistics
    stats = {
        'anish_mean': float(np.mean(emb_anish)),
        'anish_std': float(np.std(emb_anish)),
        'sayani_mean': float(np.mean(emb_sayani)),
        'sayani_std': float(np.std(emb_sayani))
    }
    
    return similarity, stats

def main():
    print("="*70)
    print("🧪 TESTING DIFFERENT PREPROCESSING OPTIONS")
    print("="*70)
    
    tests = [
        ("No CLAHE", False, 0),
        ("CLAHE clipLimit=0.5", True, 0.5),
        ("CLAHE clipLimit=1.0", True, 1.0),
        ("CLAHE clipLimit=1.5", True, 1.5),
        ("CLAHE clipLimit=2.0", True, 2.0),
        ("CLAHE clipLimit=2.5", True, 2.5),
        ("CLAHE clipLimit=3.0 (current)", True, 3.0),
        ("CLAHE clipLimit=4.0", True, 4.0),
    ]
    
    results = []
    
    for name, clahe_enabled, clip_limit in tests:
        print(f"\n📊 Test: {name}")
        try:
            similarity, stats = test_with_settings(clahe_enabled, clip_limit)
            print(f"   Similarity: {similarity:.4f} ({similarity*100:.1f}%)")
            print(f"   Anish embedding: mean={stats['anish_mean']:.4f}, std={stats['anish_std']:.4f}")
            print(f"   Sayani embedding: mean={stats['sayani_mean']:.4f}, std={stats['sayani_std']:.4f}")
            results.append((name, similarity))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((name, None))
    
    print("\n" + "="*70)
    print("📈 RESULTS SUMMARY")
    print("="*70)
    
    for name, sim in results:
        if sim is not None:
            # CORRECTED: Lower similarity between different people is BETTER
            status = "✅ EXCELLENT" if sim < 0.60 else ("✅ GOOD" if sim < 0.80 else ("⚠️  OK" if sim < 0.90 else "🚨 BAD"))
            print(f"{status}  {name}: {sim*100:.1f}%")
        else:
            print(f"❌ ERROR  {name}")
    
    # Find best setting (LOWEST similarity is best for discrimination)
    valid_results = [(n, s) for n, s in results if s is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x[1])  # Minimum similarity
        print(f"\n🏆 BEST SETTING: {best[0]} ({best[1]*100:.1f}% similarity)")
        print(f"   This provides the best discrimination between Anish and Sayani")
        
        if best[1] < 0.60:
            print("   ✅ EXCELLENT - Should work perfectly!")
        elif best[1] < 0.80:
            print("   ✅ GOOD - Should work well in most cases")
        elif best[1] < 0.90:
            print("   ⚠️  OK - May have some false rejections")
        else:
            print("   🚨 POOR - Discrimination is too low, consider alternative approaches")
    
    print("="*70)
    print("\n💡 NOTE: Lower similarity = better discrimination between different people")

if __name__ == "__main__":
    main()