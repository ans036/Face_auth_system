"""
Voice Embedder Module
Uses SpeechBrain ECAPA-TDNN for speaker verification.

Features:
- 192-dimensional speaker embeddings
- 0.8% EER on VoxCeleb benchmark
- Supports WAV and MP3 audio files
"""

import numpy as np
import os
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Global flag for SpeechBrain availability
SPEECHBRAIN_AVAILABLE = False
_voice_model = None

try:
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
    print("✅ SpeechBrain available for voice recognition")
except ImportError as e:
    print(f"⚠️ SpeechBrain not available: {e}")
    print("   Voice verification will be disabled, using face-only mode")


class VoiceEmbedder:
    """
    Generates speaker embeddings using SpeechBrain ECAPA-TDNN.
    
    Falls back gracefully if SpeechBrain is not installed.
    """
    
    def __init__(self, model_dir: str = None):
        self.model = None
        # Use home directory for cache if running as non-root
        if model_dir is None:
            home = os.path.expanduser("~")
            model_dir = os.path.join(home, ".cache", "speechbrain")
        self.model_dir = model_dir
        self.embedding_dim = 192
        self.sample_rate = 16000
        
        if SPEECHBRAIN_AVAILABLE:
            self._init_model()
    
    def _init_model(self):
        """Initialize the SpeechBrain ECAPA-TDNN model."""
        try:
            print("🎤 Loading SpeechBrain ECAPA-TDNN model...")
            
            # Create model directory if needed
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Load pre-trained model from HuggingFace
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.model_dir,
                run_opts={"device": "cpu"}
            )
            
            print("✅ Voice embedder initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to load voice model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if voice embedding is available."""
        return self.model is not None
    
    def load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to WAV or MP3 file
            
        Returns:
            Preprocessed audio tensor or None if failed
        """
        if not SPEECHBRAIN_AVAILABLE:
            return None
            
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            return waveform
            
        except Exception as e:
            print(f"⚠️ Failed to load audio {audio_path}: {e}")
            return None
    
    def embed(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Generate speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            192-dimensional L2-normalized embedding or None
        """
        if not self.is_available():
            return None
        
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        try:
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)
            
            # Convert to numpy and flatten
            emb = embedding.squeeze().cpu().numpy()
            
            # L2 normalize
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            return emb.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Failed to generate voice embedding: {e}")
            return None
    
    def embed_from_bytes(self, audio_bytes: bytes, format: str = "wav") -> Optional[np.ndarray]:
        """
        Generate embedding from audio bytes (for API uploads).
        
        OPTIMIZED: Uses BytesIO for WAV/MP3 to avoid temp file I/O.
        Only uses temp files when ffmpeg conversion is required (webm).
        
        Args:
            audio_bytes: Raw audio data
            format: Audio format (wav, mp3, webm)
            
        Returns:
            192-dimensional embedding or None
        """
        if not self.is_available():
            return None
        
        import io
        
        # For WAV/MP3, try in-memory loading first (avoids temp file I/O)
        if format.lower() in ("wav", "mp3"):
            try:
                audio_buffer = io.BytesIO(audio_bytes)
                waveform, sample_rate = torchaudio.load(audio_buffer)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample to 16kHz if needed
                if sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                    waveform = resampler(waveform)
                
                # Generate embedding
                with torch.no_grad():
                    embedding = self.model.encode_batch(waveform)
                
                emb = embedding.squeeze().cpu().numpy()
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                
                return emb.astype(np.float32)
                
            except Exception as e:
                print(f"⚠️ In-memory audio loading failed, falling back to temp file: {e}")
                # Fall through to temp file method
        
        # For webm or fallback: use temp files (ffmpeg conversion required)
        temp_path = None
        converted_path = None
        
        try:
            import tempfile
            import subprocess
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            processing_path = temp_path
            
            # Convert WebM to WAV if needed
            if format.lower() == "webm":
                 wav_path = temp_path + ".wav"
                 try:
                     # ffmpeg -y -i input.webm -ar 16000 -ac 1 output.wav
                     cmd = ["ffmpeg", "-y", "-i", temp_path, "-ar", "16000", "-ac", "1", wav_path]
                     result = subprocess.run(cmd, capture_output=True, text=True)
                     
                     if result.returncode != 0:
                         print(f"⚠️ WebM conversion failed (code {result.returncode}):\n{result.stderr}")
                     else:
                         processing_path = wav_path
                         converted_path = wav_path
                 except Exception as e:
                     print(f"⚠️ WebM conversion error: {e}")
            
            # Generate embedding
            emb = self.embed(processing_path)
            
            return emb
            
        except Exception as e:
            print(f"⚠️ Failed to process audio bytes: {e}")
            return None
            
        finally:
            # Clean up
            if temp_path and os.path.exists(temp_path):
                try: 
                    os.unlink(temp_path)
                except: 
                    pass
            if converted_path and os.path.exists(converted_path):
                try: 
                    os.unlink(converted_path)
                except:
                    pass
    
    def verify(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare two voice embeddings.
        
        Args:
            emb1, emb2: Voice embeddings to compare
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Cosine similarity (embeddings are already L2 normalized)
        similarity = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, similarity))


# Singleton instance
_voice_embedder = None

def get_voice_embedder() -> VoiceEmbedder:
    """Get or create the voice embedder singleton."""
    global _voice_embedder
    if _voice_embedder is None:
        _voice_embedder = VoiceEmbedder()
    return _voice_embedder
