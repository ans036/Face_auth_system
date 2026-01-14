from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.embedder import FaceEmbedder
from core.verifier import verify
from db.crud import get_user
from utils.image import read_image
import numpy as np

router = APIRouter()
embedder = FaceEmbedder()

@router.post("/")
async def authenticate(username: str = Form(...), file: UploadFile = File(...)):
    try:
        user = get_user(username)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        content = await file.read()
        img = read_image(content)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        probe = embedder.embed(img)
        probe = np.asarray(probe, dtype=np.float32)
        success, score = verify(user.embedding, probe, float(0.15))
        return {"authenticated": bool(success), "score": float(score)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
