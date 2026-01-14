from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.embedder import FaceEmbedder
from db.crud import create_user
from utils.image import read_image

import numpy as np

router = APIRouter()
embedder = FaceEmbedder()

@router.post("/")
async def enroll(username: str = Form(...), file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = read_image(content)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        embedding = embedder.embed(img)
        embedding = np.asarray(embedding, dtype=np.float32)
        create_user(username, embedding)
        return {"status": "enrolled", "user": username}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
