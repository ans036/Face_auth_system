# Biometric Evaluation Report

## Summary

| Metric | Face Recognition | Voice Verification |
|--------|------------------|-------------------|
| **EER** | 18.89% | 21.68% |
| EER Threshold | 0.041 | 0.058 |
| AUC | 0.8914 | 0.8678 |
| FAR @ 1% FRR | 100.00% | 100.00% |
| FAR @ 10% FRR | 100.00% | 100.00% |

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
