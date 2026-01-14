import onnxruntime as ort
import numpy as np
import cv2
import os

MODEL_PATH = "/app/models/arcface.onnx"

class FaceEmbedder:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}. Run models/download_model.sh")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, img):
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return np.expand_dims(img, 0)

    def embed(self, img):
        # 1. Resize the image to 112x112
        img = cv2.resize(img, (112, 112))
        
        # 2. Convert to float32 and normalize (standard for ArcFace)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        
        # 3. Add the Batch dimension (1, 112, 112, 3)
        # NOTE: DO NOT use .transpose(2, 0, 1) here
        inp = np.expand_dims(img, axis=0)
        
        # 4. Run inference
        emb = self.session.run([self.output_name], {self.input_name: inp})[0][0]
        return emb
