# Voice Accuracy Production Improvements

## Comprehensive Research & Implementation Guide

This document details evidence-based techniques to improve voice speaker verification accuracy to production-grade levels, based on state-of-the-art research and analysis of the current codebase.

---

## Current State Analysis

### What We Have

| Component | Current Implementation | Notes |
|-----------|----------------------|-------|
| Model | SpeechBrain ECAPA-TDNN | 0.8% EER on VoxCeleb benchmark |
| Embedding | 192-dimensional | L2-normalized |
| Preprocessing | Minimal | Only resampling to 16kHz |
| Scoring | Cosine similarity | Direct dot product |
| Fusion Weight | 15% | Voice is secondary modality |

### Identified Gaps

1. **No noise reduction** before embedding - noisy environments degrade accuracy
2. **No Voice Activity Detection (VAD)** - silence/noise included in embedding
3. **Single-sample enrollment** - prone to session-specific artifacts
4. **No quality-based rejection** - poor recordings still processed
5. **No score normalization** - raw scores not calibrated to impostor distribution

---

## Production-Grade Improvements

### 1. Audio Preprocessing Pipeline

#### 1.1 Voice Activity Detection (VAD)

**Why:** Non-speech segments (silence, background noise) dilute speaker-specific information.

**Implementation:**

```python
from silero_vad import Silero_VAD

class VoicePreprocessor:
    def __init__(self):
        self.vad = Silero_VAD(threshold=0.5)
        self.min_speech_duration = 1.5  # seconds
    
    def extract_speech(self, waveform: torch.Tensor, sr: int = 16000):
        """Remove non-speech segments."""
        speech_timestamps = self.vad.get_speech_timestamps(waveform, sr)
        
        # Concatenate speech segments
        speech_chunks = []
        for segment in speech_timestamps:
            speech_chunks.append(waveform[segment['start']:segment['end']])
        
        if not speech_chunks:
            return None  # No speech detected
        
        return torch.cat(speech_chunks)
```

**Expected improvement:** 5-10% relative EER reduction in noisy conditions.

#### 1.2 Spectral Noise Reduction

**Why:** Environmental noise masks speaker-discriminative features.

**Options:**

| Method | Complexity | Quality |
|--------|------------|---------|
| **Spectral Subtraction** | Low | Moderate |
| **Wiener Filtering** | Medium | Good |
| **WebRTC NS** | Low | Good (optimized) |
| **Deep Denoising** | High | Excellent |

**Recommended:** WebRTC noise suppression (already optimized for voice)

```python
import webrtcvad  # or noisereduce library

def denoise_audio(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Apply spectral noise reduction."""
    import noisereduce as nr
    return nr.reduce_noise(y=waveform, sr=sr, stationary=True)
```

#### 1.3 Audio Quality Assessment

**Why:** Reject recordings that will produce unreliable embeddings.

**Quality checks:**

```python
def assess_audio_quality(waveform: np.ndarray, sr: int = 16000):
    """Return quality score and rejection reason."""
    
    # 1. Duration check
    duration = len(waveform) / sr
    if duration < 1.5:
        return 0.0, "Audio too short (min 1.5s)"
    
    # 2. Signal level check
    rms = np.sqrt(np.mean(waveform**2))
    if rms < 0.001:
        return 0.1, "Audio too quiet"
    
    # 3. Clipping detection
    clipping_ratio = np.mean(np.abs(waveform) > 0.99)
    if clipping_ratio > 0.01:
        return 0.2, "Audio is clipped"
    
    # 4. Signal-to-Noise Ratio estimate
    snr = estimate_snr(waveform, sr)
    if snr < 10:  # dB
        return 0.3, f"Low SNR ({snr:.1f} dB)"
    
    return 1.0, None  # Accept
```

---

### 2. Enhanced Model Training (If Fine-tuning)

#### 2.1 Data Augmentation

**Research shows:** Extensive augmentation is key to robustness.

**Recommended augmentations:**

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| **Noise Addition** | MUSAN dataset, SNR 5-20dB | Noise robustness |
| **Room Reverberation** | RIR datasets | Channel robustness |
| **Speed Perturbation** | 0.9x - 1.1x | Speaking rate variation |
| **SpecAugment** | freq=10, time=5 | Feature masking |

**Implementation with SpeechBrain:**

```python
from speechbrain.lobes.augment import Augmenter

augmenter = Augmenter(
    parallel_augment=True,
    concat_original=True,
    augmentations={
        "noise": {"snr_low": 5, "snr_high": 20, "csv_file": "musan.csv"},
        "reverb": {"rir_path": "/data/rirs_noises/"},
        "speed": {"speeds": [90, 100, 110]},
    }
)
```

#### 2.2 Loss Function

**Current:** Pre-trained on AAM-Softmax (Additive Angular Margin)

**For fine-tuning:** Use AAM-Softmax with proper margins:

```python
# SpeechBrain AAM-Softmax parameters
margin = 0.2
scale = 30
```

---

### 3. Multi-Sample Enrollment

#### 3.1 Enrollment Protocol

**Problem:** Single recording captures one speaking style/environment.

**Solution:** Require 3-5 samples with diversity:

```python
class EnrollmentManager:
    def __init__(self, min_samples: int = 3, min_quality: float = 0.7):
        self.min_samples = min_samples
        self.min_quality = min_quality
    
    def enroll_voice(self, username: str, audio_samples: List[bytes]):
        """Enroll with multiple samples for robust representation."""
        if len(audio_samples) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples")
        
        embeddings = []
        for audio in audio_samples:
            # Quality check
            quality, reason = assess_audio_quality(audio)
            if quality < self.min_quality:
                continue  # Skip low-quality samples
            
            # Preprocess and embed
            processed = preprocess(audio)
            emb = voice_embedder.embed(processed)
            embeddings.append(emb)
        
        if len(embeddings) < self.min_samples:
            raise ValueError("Not enough high-quality samples")
        
        # Store all embeddings (or store centroid)
        for emb in embeddings:
            create_voice_entry(username, emb)
```

#### 3.2 Embedding Aggregation Strategies

| Strategy | Method | Trade-off |
|----------|--------|-----------|
| **Store All** | Keep each embedding | Best accuracy, more storage |
| **Centroid** | Average embeddings | Compact, slightly lower accuracy |
| **PLDA** | Probabilistic scoring | Best, but complex |

**Recommended:** Store all embeddings, average top-K scores during verification.

---

### 4. Score Normalization

#### 4.1 S-Norm (Symmetric Normalization)

**Why:** Raw cosine similarity varies across speakers and conditions.

**Method:**

```python
def snorm_score(probe_emb, enrolled_embs, cohort_embs):
    """
    S-norm: Normalize scores using cohort statistics.
    
    cohort_embs: 100-300 diverse speaker embeddings (not enrolled)
    """
    # Raw score
    raw_score = np.dot(probe_emb, enrolled_embs.mean(axis=0))
    
    # Probe-centric normalization
    probe_cohort_scores = [np.dot(probe_emb, c) for c in cohort_embs]
    z_probe = (raw_score - np.mean(probe_cohort_scores)) / np.std(probe_cohort_scores)
    
    # Claim-centric normalization
    claim_centroid = enrolled_embs.mean(axis=0)
    claim_cohort_scores = [np.dot(claim_centroid, c) for c in cohort_embs]
    z_claim = (raw_score - np.mean(claim_cohort_scores)) / np.std(claim_cohort_scores)
    
    # Symmetric combination
    return 0.5 * (z_probe + z_claim)
```

**Expected improvement:** 10-20% relative EER reduction.

#### 4.2 Adaptive Threshold (Like Face Module)

Port the adaptive threshold logic from `multimodal.py`:

```python
def calibrate_voice_threshold(self):
    """Calculate adaptive voice threshold from gallery."""
    # Inter-class (different speakers)
    # Intra-class (same speaker)
    # Set threshold similar to face module
```

---

### 5. Production Deployment Considerations

#### 5.1 Environment Conditioning

| Environment | Challenge | Mitigation |
|-------------|-----------|------------|
| **Office** | Background chatter | VAD + noise reduction |
| **Mobile** | Variable mics | Gain normalization |
| **Outdoor** | Wind/traffic | High SNR requirement |
| **Call center** | Phone quality | Band-pass filter |

#### 5.2 Failure Modes and Fallbacks

```python
class VoiceAuthenticator:
    def verify(self, audio: bytes, claimed_id: str) -> AuthResult:
        # Quality gate
        quality, reason = self.assess_quality(audio)
        if quality < 0.5:
            return AuthResult(
                success=False,
                fallback="face_only",
                message=f"Voice quality insufficient: {reason}"
            )
        
        # Normal verification
        ...
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. ✅ Add VAD to remove silence
2. ✅ Add basic quality score with rejection threshold
3. ✅ Require minimum audio duration (1.5s)

### Phase 2: Moderate Effort (3-5 days)
1. Implement spectral noise reduction
2. Add multi-sample enrollment (minimum 3 samples)
3. Implement score averaging across enrollments

### Phase 3: Advanced (1-2 weeks)
1. Implement S-norm score normalization
2. Add cohort model for normalization
3. Fine-tune ECAPA-TDNN on domain-specific data (if available)

---

## Expected Accuracy Improvements

| Enhancement | Relative EER Reduction |
|-------------|----------------------|
| VAD preprocessing | 5-10% |
| Noise reduction | 10-15% |
| Multi-sample enrollment | 15-20% |
| Score normalization | 10-20% |
| **Combined** | **30-50%** |

**Target:** Reduce production EER from ~2-3% (noisy conditions) to <1%.

---

## References

1. Desplanques et al., "ECAPA-TDNN" (Interspeech 2020)
2. Snyder et al., "X-Vectors: Robust DNN Embeddings" (ICASSP 2018)
3. Matějka et al., "Analysis of Score Normalization" (Interspeech 2017)
4. SpeechBrain VoxCeleb recipe documentation
5. NIST SRE evaluation protocols
