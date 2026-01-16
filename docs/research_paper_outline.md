# Multi-Modal Biometric Authentication: Adaptive Fusion of Face, Voice, and Liveness Detection

## Research Paper Structure (IEEE Conference Format)

---

## Abstract

> A ~150-word summary covering:
> - Problem: Current biometric systems are vulnerable to spoofing, single-modality failures, and threshold tuning issues
> - Solution: Multi-modal fusion with adaptive threshold calibration and softmax classification
> - Key results: [To be filled after experiments] Recognition accuracy, FAR/FRR, latency metrics
> - Significance: Production-ready system with novel accuracy-preserving optimizations

---

## 1. Introduction

### 1.1 Problem Statement
- Limitations of single-modal biometric authentication
- Presentation attacks (photos, videos, masks)
- Static threshold issues in varying enrollment sizes
- Trade-offs between security (FAR) and usability (FRR)

### 1.2 Contributions
1. **Adaptive Threshold Calibration**: Automatic threshold adjustment based on gallery statistics
2. **Softmax Classification**: Probabilistic interpretation of multi-modal scores
3. **Score-Level Fusion**: Weighted combination of face and voice modalities
4. **Production-Ready Implementation**: PostgreSQL + pgvector for scalable vector search

---

## 2. Related Work

### 2.1 Face Recognition
- **ArcFace / InsightFace**: Angular margin softmax losses
- **Buffalo-L model pack**: State-of-the-art accuracy on LFW/CFP-FP benchmarks
- Cite: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)

### 2.2 Speaker Verification
- **ECAPA-TDNN**: Enhanced TDNN architecture with SE blocks and attention
- **SpeechBrain framework**: Open-source toolkit for speaker verification
- Cite: Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention..." (Interspeech 2020)

### 2.3 Multi-Modal Biometrics
- Score-level vs feature-level fusion
- Information fusion theory (Bayes, Dempster-Shafer)
- Cite: Ross & Jain, "Information fusion in biometrics" (Pattern Recognition Letters 2003)

### 2.4 Liveness Detection
- Eye Aspect Ratio (EAR) for blink detection
- Cite: Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks" (2016)
- Remote photoplethysmography (rPPG) advances
- Deep learning approaches for presentation attack detection

---

## 3. System Architecture

### 3.1 Overview Diagram
```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│  WebRTC     │───▶│  FastAPI     │───▶│  PostgreSQL    │
│  (Camera)   │    │  Backend     │    │  + pgvector    │
└─────────────┘    └──────────────┘    └────────────────┘
      │                   │
      │            ┌──────┴──────┐
      │            │             │
┌─────▼─────┐  ┌───▼────┐  ┌─────▼─────┐
│ Liveness  │  │ Face   │  │  Voice    │
│ (EAR/Blink)│  │ (512-d)│  │  (192-d)  │
└───────────┘  └────────┘  └───────────┘
      │            │             │
      └────────────┼─────────────┘
                   ▼
           ┌──────────────┐
           │  Multimodal  │
           │   Fusion     │
           └──────────────┘
```

### 3.2 Face Embedding Module
- InsightFace Buffalo-L (w600k_r50)
- 512-dimensional L2-normalized embeddings
- Detection + alignment + recognition pipeline

### 3.3 Voice Embedding Module  
- SpeechBrain ECAPA-TDNN
- 192-dimensional speaker embeddings
- 16kHz resampling, mono conversion

### 3.4 Liveness Detection
- MediaPipe FaceMesh (468 landmarks)
- Eye Aspect Ratio (EAR) calculation
- Temporal blink pattern recognition

---

## 4. Proposed Methods

### 4.1 Adaptive Threshold Calibration

**Algorithm:**
1. Load all gallery embeddings at startup
2. Calculate inter-class similarities (different users)
3. Calculate intra-class similarities (same user)
4. Set threshold = max(inter-class) + margin

```
threshold = max(max_inter + 0.03, mean_inter + 2σ_inter)
threshold = clamp(threshold, 0.40, 0.65)
```

**Advantages:**
- No manual threshold tuning required
- Adapts to gallery size and embedding distribution
- Prevents false acceptances as gallery grows

### 4.2 Softmax Classification for Identity Decisions

**Motivation:** Raw cosine similarity lacks interpretability. Convert to probabilities.

**Method:**
```python
P(user_i) = exp(score_i / τ) / Σ exp(score_j / τ)
```

Where τ (temperature) = 0.15 for sharp distributions.

**Decision Rules:**
1. Reject if P(Unknown) > 25%
2. Accept if P(best_user) ≥ 60% AND face_score ≥ 0.35
3. Voice boost: accept borderline if voice confirms

### 4.3 Multimodal Score Fusion

**Weighted combination:**
```
combined = 0.85 × face_score + 0.15 × voice_score
```

**Adaptive voice contribution:**
- Voice only included if voice_score > 0.10
- Voice assists but never blocks authentication
- Handles missing modality gracefully

---

## 5. Experimental Setup

### 5.1 Datasets
- **Face**: [Specify dataset, e.g., LFW, custom enrolled users]
- **Voice**: [Specify dataset, e.g., VoxCeleb subset, custom recordings]
- **Test protocol**: N genuine pairs, M impostor pairs

### 5.2 Evaluation Metrics
| Metric | Description |
|--------|-------------|
| FAR | False Acceptance Rate |
| FRR | False Rejection Rate |
| EER | Equal Error Rate |
| TAR@1%FAR | True Accept Rate at 1% FAR |
| Latency | End-to-end identification time |

### 5.3 Baseline Comparisons
- Face-only (fixed threshold)
- Face-only (adaptive threshold)
- Face + Voice fusion
- Face + Voice + Liveness

---

## 6. Results

### 6.1 Recognition Accuracy
[Table: FAR, FRR, EER for each configuration]

### 6.2 Liveness Detection Effectiveness
[Attack success rates for photo, video, replay attacks]

### 6.3 Performance Benchmarks
[Bar chart: SQLite vs PostgreSQL latency]
[Line chart: Query time vs gallery size]

### 6.4 Ablation Study
- Impact of adaptive threshold vs fixed
- Impact of softmax temperature
- Impact of voice weight

---

## 7. Discussion

### 7.1 Key Findings
- Adaptive thresholds reduce FAR by X% with minimal FRR increase
- Softmax classification improves borderline case decisions
- pgvector provides 100x speedup at scale

### 7.2 Limitations
- Blink-only liveness vulnerable to video attacks
- Voice requires quiet environment
- Single-sample enrollment less robust

### 7.3 Future Work
- Deep learning-based anti-spoofing (rPPG, texture analysis)
- Vision Transformers for face embedding
- On-device deployment optimization

---

## 8. Conclusion

Summary of contributions and practical impact.

---

## References

```bibtex
@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech},
  year={2020}
}

@article{soukupova2016eye,
  title={Real-Time Eye Blink Detection using Facial Landmarks},
  author={Soukupová, Tereza and Čech, Jan},
  journal={CVWW},
  year={2016}
}

@article{ross2003information,
  title={Information fusion in biometrics},
  author={Ross, Arun and Jain, Anil},
  journal={Pattern Recognition Letters},
  year={2003}
}
```

---

## Appendix A: System Configuration

| Component | Specification |
|-----------|--------------|
| Face Model | InsightFace Buffalo-L (w600k_r50) |
| Voice Model | SpeechBrain ECAPA-TDNN (VoxCeleb) |
| Database | PostgreSQL 15 + pgvector 0.5 |
| Index Type | HNSW (m=16, ef_construction=64) |
| Framework | FastAPI + SQLAlchemy |

## Appendix B: API Specification

See OpenAPI documentation at `/docs`.
