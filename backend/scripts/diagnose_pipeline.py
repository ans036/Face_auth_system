#!/usr/bin/env python3
"""
Diagnostic script to trace the face recognition pipeline and identify the bug
causing high similarity between different people.
"""

import sys
sys.path.insert(0, '/app')

import cv2
import numpy as np
import os
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from utils.image import crop_box
from utils.allignment import align_face

def save_debug_image(img, name, step):
    """Save intermediate images for debugging"""
    debug_dir = "/app/debug_output"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Convert RGB to BGR for OpenCV saving
    if len(img.shape) == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    else:
        img_bgr = img
    
    filepath = os.path.join(debug_dir, f"{name}_{step}.jpg")
    cv2.imwrite(filepath, img_bgr)
    print(f"   📸 Saved: {filepath}")

def apply_clahe(img, clip_limit=4.0):
    """Apply CLAHE preprocessing"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    img_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

def diagnose_image(img_path, name, detector, embedder):
    """Process image step by step with debugging"""
    print(f"\n{'='*60}")
    print(f"📊 DIAGNOSING: {name}")
    print(f"   Path: {img_path}")
    print(f"{'='*60}")
    
    # Step 1: Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"   ❌ Failed to load image!")
        return None
    print(f"   ✅ Step 1: Loaded image - Shape: {img_bgr.shape}")
    save_debug_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), name, "1_original")
    
    # Step 2: Apply CLAHE
    img_clahe = apply_clahe(img_bgr, clip_limit=4.0)
    print(f"   ✅ Step 2: Applied CLAHE (clipLimit=4.0)")
    save_debug_image(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB), name, "2_clahe")
    
    # Step 3: Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
    print(f"   ✅ Step 3: Converted BGR to RGB")
    
    # Step 4: Detect face
    box, kps = detector.detect(img_rgb)
    if box is None:
        print(f"   ❌ Step 4: No face detected!")
        return None
    print(f"   ✅ Step 4: Face detected - Box: {box}")
    print(f"              Keypoints available: {len(kps) if kps else 0}")
    
    # Draw bounding box for visualization
    img_with_box = img_rgb.copy()
    y1, x1, y2, x2 = [int(v) for v in box]
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
    save_debug_image(img_with_box, name, "4_detected")
    
    # Step 5: Align face
    aligned = align_face(img_rgb, kps)
    print(f"   ✅ Step 5: Aligned face - Shape: {aligned.shape}")
    save_debug_image(aligned, name, "5_aligned")
    
    # Step 6: Crop face
    face = crop_box(aligned, box)
    print(f"   ✅ Step 6: Cropped face - Shape: {face.shape}")
    save_debug_image(face, name, "6_cropped")
    
    # Step 7: Resize to 112x112 for embedding
    face_resized = cv2.resize(face, (112, 112))
    print(f"   ✅ Step 7: Resized face - Shape: {face_resized.shape}")
    save_debug_image(face_resized, name, "7_resized")
    
    # Step 8: Generate embedding
    emb = embedder.embed(face)
    print(f"   ✅ Step 8: Generated embedding - Shape: {emb.shape}")
    print(f"              Embedding stats: min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}, std={emb.std():.4f}")
    print(f"              L2 norm: {np.linalg.norm(emb):.4f}")
    
    return emb, face_resized

def main():
    print("="*70)
    print("🔍 FACE RECOGNITION PIPELINE DIAGNOSTIC")
    print("="*70)
    
    detector = FaceDetector()
    embedder = FaceEmbedder()
    
    # Process Anish
    result_anish = diagnose_image("/app/database/Anish/pic1.jpeg", "Anish", detector, embedder)
    
    # Process Sayani
    result_sayani = diagnose_image("/app/database/Sayani/pic1.jpeg", "Sayani", detector, embedder)
    
    if result_anish is None or result_sayani is None:
        print("\n❌ Cannot compare - one or both images failed processing")
        return
    
    emb_anish, face_anish = result_anish
    emb_sayani, face_sayani = result_sayani
    
    # Compare embeddings
    print("\n" + "="*70)
    print("📈 EMBEDDING COMPARISON")
    print("="*70)
    
    # Cosine similarity
    similarity = float(np.dot(emb_anish, emb_sayani))
    print(f"   Cosine Similarity: {similarity:.4f} ({similarity*100:.1f}%)")
    
    # Euclidean distance
    euclidean = float(np.linalg.norm(emb_anish - emb_sayani))
    print(f"   Euclidean Distance: {euclidean:.4f}")
    
    # Element-wise correlation
    correlation = float(np.corrcoef(emb_anish, emb_sayani)[0, 1])
    print(f"   Pearson Correlation: {correlation:.4f}")
    
    # Check if embeddings are nearly identical
    diff = np.abs(emb_anish - emb_sayani)
    print(f"\n   Embedding difference stats:")
    print(f"      Min diff: {diff.min():.6f}")
    print(f"      Max diff: {diff.max():.6f}")
    print(f"      Mean diff: {diff.mean():.6f}")
    print(f"      Stddev diff: {diff.std():.6f}")
    
    # Count how many elements are nearly identical
    threshold = 0.01
    identical_count = np.sum(diff < threshold)
    print(f"      Elements with diff < {threshold}: {identical_count}/{len(diff)} ({100*identical_count/len(diff):.1f}%)")
    
    # Check if the cropped faces are similar
    print("\n" + "="*70)
    print("📷 CROPPED FACE COMPARISON")
    print("="*70)
    
    # Resize both to same size for pixel comparison
    face_a = cv2.resize(face_anish, (112, 112)).astype(np.float32)
    face_s = cv2.resize(face_sayani, (112, 112)).astype(np.float32)
    
    # Pixel-wise SSIM approximation
    face_similarity = 1 - np.mean(np.abs(face_a - face_s)) / 255.0
    print(f"   Pixel-wise similarity: {face_similarity:.4f} ({face_similarity*100:.1f}%)")
    
    # Histogram comparison
    hist_a = cv2.calcHist([face_anish], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([face_sayani], [0], None, [256], [0, 256])
    hist_correlation = cv2.compareHist(hist_a, hist_s, cv2.HISTCMP_CORREL)
    print(f"   Histogram correlation: {hist_correlation:.4f}")
    
    print("\n" + "="*70)
    print("🎯 DIAGNOSIS SUMMARY")
    print("="*70)
    
    if similarity > 0.90:
        print("   🚨 CRITICAL: Embeddings are too similar (>90%)!")
        
        if face_similarity > 0.90:
            print("   → Problem: Cropped faces are nearly identical")
            print("   → Likely cause: Face detection/cropping returning same region")
        else:
            print("   → Problem: Different faces produce similar embeddings")
            print("   → Likely cause: Model not generating discriminative features")
            print("   → Solution: Try different face recognition model")
    else:
        print("   ✅ Embeddings are sufficiently different")
    
    print("\n📁 Debug images saved to /app/debug_output/")
    print("="*70)

if __name__ == "__main__":
    main()
