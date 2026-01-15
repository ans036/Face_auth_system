"""
Database Models with pgvector support for vector similarity search.

Uses pgvector's Vector type for efficient similarity queries in PostgreSQL.
Falls back to LargeBinary for SQLite compatibility during development.
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, LargeBinary, Float, Index
import os

Base = declarative_base()

# Check if we're using PostgreSQL (pgvector available)
DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_PGVECTOR = DATABASE_URL.startswith("postgresql")

if USE_PGVECTOR:
    try:
        from pgvector.sqlalchemy import Vector
        print("✅ pgvector extension loaded for PostgreSQL")
    except ImportError:
        print("⚠️ pgvector not available, using LargeBinary fallback")
        USE_PGVECTOR = False


class User(Base):
    """User account with face embedding."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    
    # Use Vector type for PostgreSQL, LargeBinary for SQLite
    if USE_PGVECTOR:
        embedding = Column(Vector(512), nullable=False)
    else:
        embedding = Column(LargeBinary, nullable=False)


class Gallery(Base):
    """
    Face embeddings gallery.
    
    With pgvector, similarity searches are done natively in PostgreSQL:
    SELECT username, 1 - (embedding <=> query) as similarity
    FROM gallery
    ORDER BY embedding <=> query
    LIMIT 5;
    """
    __tablename__ = "gallery"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False, index=True)
    
    if USE_PGVECTOR:
        embedding = Column(Vector(512), nullable=False)
    else:
        embedding = Column(LargeBinary, nullable=False)


class VoiceGallery(Base):
    """
    Voice embeddings gallery for multi-modal auth.
    
    ECAPA-TDNN produces 192-dimensional embeddings.
    """
    __tablename__ = "voice_gallery"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False, index=True)
    
    if USE_PGVECTOR:
        embedding = Column(Vector(192), nullable=False)
    else:
        embedding = Column(LargeBinary, nullable=False)


# Create HNSW indexes for fast approximate nearest neighbor search
# These will be created after tables exist
def create_vector_indexes(engine):
    """Create HNSW indexes for vector similarity search (PostgreSQL only)."""
    if not USE_PGVECTOR:
        return
    
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        
        # Create HNSW index for face gallery (cosine distance)
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS gallery_embedding_idx 
                ON gallery 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            conn.commit()
            print("✅ Created HNSW index for face gallery")
        except Exception as e:
            print(f"⚠️ Could not create face gallery index: {e}")
        
        # Create HNSW index for voice gallery
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS voice_gallery_embedding_idx 
                ON voice_gallery 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            conn.commit()
            print("✅ Created HNSW index for voice gallery")
        except Exception as e:
            print(f"⚠️ Could not create voice gallery index: {e}")
