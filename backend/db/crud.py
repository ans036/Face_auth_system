import numpy as np
from db.session import SessionLocal
from db.models import User, Gallery, VoiceGallery

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
            out.append({"username": r.username, "embedding": r.embedding})
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

def get_gallery_stats():
    """Get count of embeddings per user."""
    s = SessionLocal()
    try:
        rows = s.query(Gallery).all()
        stats = {}
        for r in rows:
            if r.username not in stats:
                stats[r.username] = 0
            stats[r.username] += 1
        return stats
    finally:
        s.close()

def delete_user_from_gallery(username: str):
    """Delete all embeddings for a specific user."""
    s = SessionLocal()
    try:
        s.query(Gallery).filter_by(username=username).delete()
        s.commit()
    finally:
        s.close()

def get_user_embeddings(username: str):
    """Get all embeddings for a specific user."""
    s = SessionLocal()
    try:
        rows = s.query(Gallery).filter_by(username=username).all()
        return [np.frombuffer(r.embedding, dtype=np.float32) for r in rows]
    finally:
        s.close()

# =====================
# Voice Gallery CRUD
# =====================

def create_voice_entry(username: str, embedding: np.ndarray):
    """Add a voice embedding to the gallery."""
    s = SessionLocal()
    try:
        arr = np.asarray(embedding, dtype=np.float32)
        v = VoiceGallery(username=username, embedding=arr.tobytes())
        s.add(v)
        s.commit()
    finally:
        s.close()

def get_voice_gallery():
    """Get all voice embeddings."""
    s = SessionLocal()
    try:
        rows = s.query(VoiceGallery).all()
        out = []
        for r in rows:
            out.append({"username": r.username, "embedding": r.embedding})
        return out
    finally:
        s.close()

def clear_voice_gallery():
    """Clear all voice embeddings."""
    s = SessionLocal()
    try:
        s.query(VoiceGallery).delete()
        s.commit()
    finally:
        s.close()

def get_voice_stats():
    """Get count of voice embeddings per user."""
    s = SessionLocal()
    try:
        rows = s.query(VoiceGallery).all()
        stats = {}
        for r in rows:
            if r.username not in stats:
                stats[r.username] = 0
            stats[r.username] += 1
        return stats
    finally:
        s.close()
