import cv2
import numpy as np

def assess_image_quality(img):
    """
    Assess image quality based on multiple metrics.
    
    Returns:
        quality_score (float): 0-1 score, higher is better
        metrics (dict): Individual quality metrics
    """
    metrics = {}
    
    # 1. Sharpness (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    metrics['sharpness'] = float(laplacian_var)
    
    # Normalize sharpness (typical range 0-1000, clip and scale to 0-1)
    sharpness_score = min(laplacian_var / 500.0, 1.0)
    
    # 2. Brightness (check if image is not too dark or too bright)
    mean_brightness = np.mean(gray)
    # Optimal brightness is around 127, penalize deviation
    brightness_deviation = abs(mean_brightness - 127) / 127
    brightness_score = max(0, 1.0 - brightness_deviation)
    metrics['brightness'] = float(mean_brightness)
    
    # 3. Contrast (standard deviation of pixel values)
    contrast = np.std(gray)
    metrics['contrast'] = float(contrast)
    # Normalize contrast (typical good range 40-80)
    contrast_score = min(contrast / 60.0, 1.0)
    
    # 4. Composite quality score (weighted average)
    quality_score = (
        0.5 * sharpness_score +      # Sharpness most important
        0.3 * brightness_score +      # Brightness moderately important
        0.2 * contrast_score          # Contrast least critical
    )
    
    metrics['quality_score'] = float(quality_score)
    
    return quality_score, metrics


def is_face_quality_acceptable(face_img, min_quality=0.3):
    """
    Check if a face image meets minimum quality standards.
    
    Args:
        face_img: Face image (RGB)
        min_quality: Minimum quality score (0-1)
    
    Returns:
        (acceptable, quality_score, reason)
    """
    h, w = face_img.shape[:2]
    
    # Check 1: Minimum resolution
    if h < 80 or w < 80:
        return False, 0.0, f"Too small: {w}x{h}"
    
    # Check 2: Image quality metrics
    quality_score, metrics = assess_image_quality(face_img)
    
    if quality_score < min_quality:
        reasons = []
        if metrics['sharpness'] < 50:
            reasons.append("blurry")
        if metrics['brightness'] < 50 or metrics['brightness'] > 200:
            reasons.append("poor lighting")
        if metrics['contrast'] < 30:
            reasons.append("low contrast")
        
        reason = ", ".join(reasons) if reasons else "low quality"
        return False, quality_score, reason
    
    return True, quality_score, "OK"


def enhance_face_image(img):
    """
    Apply quality enhancements to face image.
    
    Args:
        img: Face image in RGB
    
    Returns:
        Enhanced image
    """
    # Convert to LAB for better enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 1. CLAHE on L channel (already applied, but ensure it's there)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # 2. Mild sharpening
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    l_sharpened = cv2.filter2D(l_enhanced, -1, kernel)
    
    # Blend 70% enhanced, 30% sharpened
    l_final = cv2.addWeighted(l_enhanced, 0.7, l_sharpened, 0.3, 0)
    
    # 3. Merge back and convert to RGB
    lab_enhanced = cv2.merge((l_final, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # 4. Ensure uint8 range
    rgb_enhanced = np.clip(rgb_enhanced, 0, 255).astype(np.uint8)
    
    return rgb_enhanced