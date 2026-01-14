"""
InsightFace-based Face Embedder with buffalo_l model pack.
This provides better face recognition accuracy than the basic ArcFace ONNX model.
"""
import numpy as np
import cv2
import os

# Try to import InsightFace first, fall back to ONNX model if not available
try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  InsightFace not available, using fallback ONNX model")

# Fallback to ONNX model
if not INSIGHTFACE_AVAILABLE:
    import onnxruntime as ort

MODEL_PATH = "/app/models/arcface.onnx"

class FaceEmbedder:
    """
    Face embedder using InsightFace buffalo_l model pack.
    Falls back to basic ONNX model if InsightFace is not available.
    
    For InsightFace, this class uses the full FaceAnalysis pipeline which
    includes both detection and recognition in one step.
    """
    
    def __init__(self, model_path=MODEL_PATH, use_insightface=True):
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        self.recognition_model = None
        
        if self.use_insightface:
            self._init_insightface()
        else:
            self._init_onnx(model_path)
    
    def _init_insightface(self):
        """Initialize InsightFace FaceAnalysis with buffalo_l model pack."""
        print("🚀 Initializing InsightFace with buffalo_l model pack...")
        
        # Set up model directory
        model_dir = "/app/models/insightface"
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize FaceAnalysis with buffalo_l (best quality model)
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=model_dir,
            providers=['CPUExecutionProvider']
        )
        # Prepare for detection at various image sizes
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Also load the recognition model directly for cropped face embedding
        rec_model_path = os.path.join(model_dir, "models/buffalo_l/w600k_r50.onnx")
        if os.path.exists(rec_model_path):
            self.recognition_model = get_model(rec_model_path, providers=['CPUExecutionProvider'])
            self.recognition_model.prepare(ctx_id=0)
            print("✅ InsightFace recognition model loaded for direct embedding")
        
        print("✅ InsightFace buffalo_l initialized successfully")
        print(f"   📊 Models loaded: {list(self.app.models.keys()) if hasattr(self.app.models, 'keys') else self.app.models}")
    
    def _init_onnx(self, model_path):
        """Fallback initialization using basic ONNX model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Check expected input shape
        input_shape = self.session.get_inputs()[0].shape
        print(f"🔍 ArcFace model expects input shape: {input_shape}")
        
        if input_shape[1] == 3:
            self.channel_first = True
            print("   Model expects: (batch, channels, height, width) - NCHW")
        elif input_shape[-1] == 3:
            self.channel_first = False
            print("   Model expects: (batch, height, width, channels) - NHWC")
        else:
            print("   ⚠️  Warning: Could not determine channel order, assuming NCHW")
            self.channel_first = True

    def embed(self, img):
        """
        Generate L2-normalized face embedding from RGB image.
        
        For InsightFace: Use the recognition model directly on the cropped face.
        For ONNX: Use the basic embedding pipeline.
        
        Args:
            img: RGB image (cropped face region)
        
        Returns:
            512-dimensional L2-normalized embedding vector
        """
        if self.use_insightface:
            return self._embed_insightface(img)
        else:
            return self._embed_onnx(img)
    
    def _embed_insightface(self, img):
        """Generate embedding using InsightFace recognition model directly."""
        # InsightFace expects BGR image
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        # If we have the direct recognition model, use it
        if self.recognition_model is not None:
            # Resize to 112x112 (expected by recognition model)
            img_resized = cv2.resize(img_bgr, (112, 112))
            
            # Get embedding directly from recognition model
            emb = self.recognition_model.get_feat(img_resized)
            
            # Flatten if needed
            if len(emb.shape) > 1:
                emb = emb.flatten()
            
            # L2 normalize
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            return emb.astype(np.float32)
        
        # Fallback: try FaceAnalysis with full detection pipeline
        # This works best with full images, not cropped faces
        faces = self.app.get(img_bgr)
        
        if len(faces) == 0:
            # Try resizing to help detection
            img_resized = cv2.resize(img_bgr, (640, 640))
            faces = self.app.get(img_resized)
            
            if len(faces) == 0:
                print("   ⚠️  InsightFace: No face detected, using ONNX fallback")
                # Fall back to ONNX if InsightFace can't detect
                return self._embed_onnx_internal(img)
        
        # Get the embedding from the first detected face
        emb = faces[0].embedding
        
        # L2 normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        return emb.astype(np.float32)
    
    def _embed_onnx_internal(self, img):
        """Internal ONNX embedding for fallback when InsightFace can't detect."""
        import onnxruntime as ort
        
        if not hasattr(self, 'fallback_session'):
            model_path = "/app/models/arcface.onnx"
            if os.path.exists(model_path):
                self.fallback_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self.fallback_input_name = self.fallback_session.get_inputs()[0].name
                self.fallback_output_name = self.fallback_session.get_outputs()[0].name
                input_shape = self.fallback_session.get_inputs()[0].shape
                self.fallback_channel_first = input_shape[1] == 3
            else:
                return np.zeros(512, dtype=np.float32)
        
        # Resize to 112x112
        img = cv2.resize(img, (112, 112))
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        
        # Transpose if needed
        if self.fallback_channel_first:
            img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        inp = np.expand_dims(img, axis=0)
        
        # Run inference
        emb = self.fallback_session.run([self.fallback_output_name], {self.fallback_input_name: inp})[0][0]
        
        # L2 normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        return emb.astype(np.float32)
    
    def _embed_onnx(self, img):
        """Embedding using basic ONNX model."""
        # 1. Resize to 112x112
        img = cv2.resize(img, (112, 112))
        
        # 2. Normalize to [-1, 1]
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        
        # 3. Transpose if model expects channel-first
        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))
        
        # 4. Add batch dimension
        inp = np.expand_dims(img, axis=0)
        
        # 5. Run inference
        emb = self.session.run([self.output_name], {self.input_name: inp})[0][0]
        
        # 6. L2 normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        return emb

    def embed_from_full_image(self, img):
        """
        Perform face detection and embedding in one step on full image.
        This is more accurate than using a separate detector + embedder.
        
        Args:
            img: Full RGB image (not cropped)
        
        Returns:
            List of dicts with 'embedding', 'bbox', 'landmarks', 'det_score' for each face
        """
        if not self.use_insightface:
            raise NotImplementedError("embed_from_full_image requires InsightFace")
        
        # Convert to BGR for InsightFace
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get all faces
        faces = self.app.get(img_bgr)
        
        results = []
        for face in faces:
            emb = face.embedding
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            landmarks = face.kps if hasattr(face, 'kps') else None
            
            results.append({
                'embedding': emb.astype(np.float32),
                'bbox': bbox,
                'landmarks': landmarks,
                'det_score': float(face.det_score)
            })
        
        return results