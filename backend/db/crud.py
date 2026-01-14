import numpy as np
from db.session import SessionLocal
from db.models import User, Gallery

def create_user(username: str, embedding: np.ndarray):
    s = SessionLocal()
    try:
        arr = np.asarray(embedding, dtype=np.float32)
        u = User(username=username, embedding=arr.tobytes())
        s.add(u)
        s.commit()
    finally:
        s.close()

def get_user(username: str):
    s = SessionLocal()
    try:
        u = s.query(User).filter_by(username=username).first()
        if u is None:
            return None
        u.embedding = np.frombuffer(u.embedding, dtype=np.float32)
        return u
    finally:
        s.close()

def create_gallery_entry(username: str, embedding: np.ndarray):
    s = SessionLocal()
    try:
        arr = np.asarray(embedding, dtype=np.float32)
        g = Gallery(username=username, embedding=arr.tobytes())
        s.add(g)
        s.commit()
    finally:
        s.close()

def get_all_gallery():
    s = SessionLocal()
    try:
        rows = s.query(Gallery).all()
        out = []
        for r in rows:
            emb = np.frombuffer(r.embedding, dtype=np.float32)
            out.append({"username": r.username, "embedding": emb})
        return out
    finally:
        s.close()

def clear_gallery():
    s = SessionLocal()
    try:
        s.query(Gallery).delete()
        s.commit()
    finally:
        s.close()
