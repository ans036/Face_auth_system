"""
Database CRUD operations with pgvector support for vector similarity search.

When using PostgreSQL+pgvector:
- Embeddings stored as native Vector type
- Similarity search done in database using HNSW index
- Much faster than in-memory Python loops

When using SQLite (fallback):
- Embeddings stored as bytes
- Similarity search done in Python
"""
import numpy as np
from db.session import SessionLocal
from db.models import User, Gallery, VoiceGallery, USE_PGVECTOR
from typing import List, Dict, Optional
from sqlalchemy import text


# =====================
# Face Gallery CRUD
# =====================

def create_user(username: str, embedding: np.ndarray):
    """Create a user with face embedding."""
    s = SessionLocal()
    try:
        arr = np.asarray(embedding, dtype=np.float32)
        if USE_PGVECTOR:
            # pgvector accepts list of floats
            u = User(username=username, embedding=arr.tolist())
        else:
            u = User(username=username, embedding=arr.tobytes())
        s.add(u)
        s.commit()
    finally:
        s.close()


def get_user(username: str):
    """Get a user by username."""
    s = SessionLocal()
    try:
        u = s.query(User).filter_by(username=username).first()
        if u is None:
            return None
        if not USE_PGVECTOR:
            u.embedding = np.frombuffer(u.embedding, dtype=np.float32)
        return u
    finally:
        s.close()


def create_gallery_entry(username: str, embedding: np.ndarray):
    """Add a face embedding to the gallery."""
    s = SessionLocal()
    try:
        arr = np.asarray(embedding, dtype=np.float32)
        if USE_PGVECTOR:
            g = Gallery(username=username, embedding=arr.tolist())
        else:
            g = Gallery(username=username, embedding=arr.tobytes())
        s.add(g)
        s.commit()
    finally:
        s.close()


def get_all_gallery() -> List[Dict]:
    """Get all face embeddings from gallery."""
    s = SessionLocal()
    try:
        rows = s.query(Gallery).all()
        out = []
        for r in rows:
            if USE_PGVECTOR:
                # pgvector returns numpy-compatible array
                emb = np.array(r.embedding, dtype=np.float32)
            else:
                emb = r.embedding  # bytes, will be parsed later
            out.append({"username": r.username, "embedding": emb})
        return out
    finally:
        s.close()


def search_similar_faces(probe_embedding: np.ndarray, limit: int = 10) -> List[Dict]:
    """
    Search for similar faces using pgvector's native similarity search.
    
    Uses cosine distance with HNSW index for fast approximate nearest neighbor.
    
    Args:
        probe_embedding: Query embedding (512-dim)
        limit: Maximum number of results
        
    Returns:
        List of {username, score, embedding_id} sorted by similarity
    """
    if not USE_PGVECTOR:
        raise NotImplementedError("Native vector search requires PostgreSQL+pgvector")
    
    s = SessionLocal()
    try:
        # Convert to list for pgvector
        probe_list = np.asarray(probe_embedding, dtype=np.float32).tolist()
        
        # Use cosine distance operator <=>
        # 1 - distance = similarity
        query = text("""
            SELECT id, username, 1 - (embedding <=> :probe) as similarity
            FROM gallery
            ORDER BY embedding <=> :probe
            LIMIT :limit
        """)
        
        result = s.execute(query, {"probe": str(probe_list), "limit": limit})
        
        matches = []
        for row in result:
            matches.append({
                "id": row.id,
                "username": row.username,
                "score": float(row.similarity)
            })
        
        return matches
    finally:
        s.close()


def search_similar_faces_by_user(probe_embedding: np.ndarray, limit_per_user: int = 5) -> Dict[str, float]:
    """
    Get per-user similarity scores using pgvector.
    
    Aggregates top-k scores per user (average of best matches).
    
    Returns:
        Dict of username -> average similarity score
    """
    if not USE_PGVECTOR:
        raise NotImplementedError("Native vector search requires PostgreSQL+pgvector")
    
    s = SessionLocal()
    try:
        probe_list = np.asarray(probe_embedding, dtype=np.float32).tolist()
        
        # Use window function to rank per user, then aggregate
        query = text("""
            WITH ranked AS (
                SELECT 
                    username,
                    1 - (embedding <=> :probe) as similarity,
                    ROW_NUMBER() OVER (PARTITION BY username ORDER BY embedding <=> :probe) as rn
                FROM gallery
            )
            SELECT username, AVG(similarity) as avg_similarity
            FROM ranked
            WHERE rn <= :limit
            GROUP BY username
            ORDER BY avg_similarity DESC
        """)
        
        result = s.execute(query, {"probe": str(probe_list), "limit": limit_per_user})
        
        return {row.username: float(row.avg_similarity) for row in result}
    finally:
        s.close()


def clear_gallery():
    """Delete all face embeddings."""
    s = SessionLocal()
    try:
        s.query(Gallery).delete()
        s.commit()
    finally:
        s.close()


def get_gallery_stats() -> Dict[str, int]:
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


def get_user_embeddings(username: str) -> List[np.ndarray]:
    """Get all embeddings for a specific user."""
    s = SessionLocal()
    try:
        rows = s.query(Gallery).filter_by(username=username).all()
        if USE_PGVECTOR:
            return [np.array(r.embedding, dtype=np.float32) for r in rows]
        else:
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
        if USE_PGVECTOR:
            v = VoiceGallery(username=username, embedding=arr.tolist())
        else:
            v = VoiceGallery(username=username, embedding=arr.tobytes())
        s.add(v)
        s.commit()
    finally:
        s.close()


def get_voice_gallery() -> List[Dict]:
    """Get all voice embeddings."""
    s = SessionLocal()
    try:
        rows = s.query(VoiceGallery).all()
        out = []
        for r in rows:
            if USE_PGVECTOR:
                emb = np.array(r.embedding, dtype=np.float32)
            else:
                emb = r.embedding
            out.append({"username": r.username, "embedding": emb})
        return out
    finally:
        s.close()


def search_similar_voices(probe_embedding: np.ndarray, limit: int = 10) -> List[Dict]:
    """Search for similar voices using pgvector."""
    if not USE_PGVECTOR:
        raise NotImplementedError("Native vector search requires PostgreSQL+pgvector")
    
    s = SessionLocal()
    try:
        probe_list = np.asarray(probe_embedding, dtype=np.float32).tolist()
        
        query = text("""
            SELECT id, username, 1 - (embedding <=> :probe) as similarity
            FROM voice_gallery
            ORDER BY embedding <=> :probe
            LIMIT :limit
        """)
        
        result = s.execute(query, {"probe": str(probe_list), "limit": limit})
        
        return [{"id": row.id, "username": row.username, "score": float(row.similarity)} for row in result]
    finally:
        s.close()


def search_similar_voices_by_user(probe_embedding: np.ndarray) -> Dict[str, float]:
    """Get per-user voice similarity scores."""
    if not USE_PGVECTOR:
        raise NotImplementedError("Native vector search requires PostgreSQL+pgvector")
    
    s = SessionLocal()
    try:
        probe_list = np.asarray(probe_embedding, dtype=np.float32).tolist()
        
        query = text("""
            SELECT username, AVG(1 - (embedding <=> :probe)) as avg_similarity
            FROM voice_gallery
            GROUP BY username
            ORDER BY avg_similarity DESC
        """)
        
        result = s.execute(query, {"probe": str(probe_list)})
        
        return {row.username: float(row.avg_similarity) for row in result}
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


def get_voice_stats() -> Dict[str, int]:
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
