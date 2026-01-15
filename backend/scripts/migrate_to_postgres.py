#!/usr/bin/env python3
"""
Migration script: SQLite -> PostgreSQL with pgvector

Migrates existing face and voice embeddings from SQLite to PostgreSQL.
Run this after starting the PostgreSQL container.

Usage:
    python scripts/migrate_to_postgres.py
"""
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def migrate_embeddings():
    """Migrate embeddings from SQLite to PostgreSQL."""
    
    # Get connection strings from environment
    sqlite_path = os.getenv("SQLITE_PATH", "/app/face_auth.db")
    postgres_url = os.getenv("DATABASE_URL", "postgresql://faceauth:faceauth_secret@postgres:5432/face_auth")
    
    if not postgres_url.startswith("postgresql"):
        print("ERROR: DATABASE_URL must be a PostgreSQL connection string")
        sys.exit(1)
    
    # Create SQLite connection
    sqlite_url = f"sqlite:///{sqlite_path}"
    print(f"Source: {sqlite_url}")
    print(f"Target: {postgres_url}")
    
    try:
        sqlite_engine = create_engine(sqlite_url)
        SQLiteSession = sessionmaker(bind=sqlite_engine)
        sqlite_session = SQLiteSession()
    except Exception as e:
        print(f"ERROR: Could not connect to SQLite: {e}")
        sys.exit(1)
    
    try:
        pg_engine = create_engine(postgres_url)
        PgSession = sessionmaker(bind=pg_engine)
        pg_session = PgSession()
    except Exception as e:
        print(f"ERROR: Could not connect to PostgreSQL: {e}")
        sys.exit(1)
    
    # Enable pgvector extension
    print("\n1. Enabling pgvector extension...")
    try:
        pg_session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        pg_session.commit()
        print("   ✓ pgvector extension enabled")
    except Exception as e:
        print(f"   ⚠ Could not enable pgvector: {e}")
    
    # Create tables if they don't exist
    print("\n2. Creating tables...")
    try:
        pg_session.execute(text("""
            CREATE TABLE IF NOT EXISTS gallery (
                id SERIAL PRIMARY KEY,
                username VARCHAR NOT NULL,
                embedding vector(512) NOT NULL
            )
        """))
        pg_session.execute(text("""
            CREATE TABLE IF NOT EXISTS voice_gallery (
                id SERIAL PRIMARY KEY,
                username VARCHAR NOT NULL,
                embedding vector(192) NOT NULL
            )
        """))
        pg_session.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR UNIQUE NOT NULL,
                embedding vector(512) NOT NULL
            )
        """))
        pg_session.commit()
        print("   ✓ Tables created")
    except Exception as e:
        print(f"   ⚠ Error creating tables: {e}")
        pg_session.rollback()
    
    # Migrate face gallery
    print("\n3. Migrating face gallery...")
    try:
        result = sqlite_session.execute(text("SELECT id, username, embedding FROM gallery"))
        rows = result.fetchall()
        
        migrated = 0
        for row in rows:
            emb = np.frombuffer(row[2], dtype=np.float32)
            emb_list = emb.tolist()
            
            pg_session.execute(
                text("INSERT INTO gallery (username, embedding) VALUES (:username, :embedding)"),
                {"username": row[1], "embedding": str(emb_list)}
            )
            migrated += 1
        
        pg_session.commit()
        print(f"   ✓ Migrated {migrated} face embeddings")
    except Exception as e:
        print(f"   ⚠ Error migrating face gallery: {e}")
        pg_session.rollback()
    
    # Migrate voice gallery
    print("\n4. Migrating voice gallery...")
    try:
        result = sqlite_session.execute(text("SELECT id, username, embedding FROM voice_gallery"))
        rows = result.fetchall()
        
        migrated = 0
        for row in rows:
            emb = np.frombuffer(row[2], dtype=np.float32)
            emb_list = emb.tolist()
            
            pg_session.execute(
                text("INSERT INTO voice_gallery (username, embedding) VALUES (:username, :embedding)"),
                {"username": row[1], "embedding": str(emb_list)}
            )
            migrated += 1
        
        pg_session.commit()
        print(f"   ✓ Migrated {migrated} voice embeddings")
    except Exception as e:
        print(f"   ⚠ Error migrating voice gallery: {e}")
        pg_session.rollback()
    
    # Create HNSW indexes
    print("\n5. Creating HNSW indexes...")
    try:
        pg_session.execute(text("""
            CREATE INDEX IF NOT EXISTS gallery_embedding_idx 
            ON gallery 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        pg_session.execute(text("""
            CREATE INDEX IF NOT EXISTS voice_gallery_embedding_idx 
            ON voice_gallery 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        pg_session.commit()
        print("   ✓ HNSW indexes created")
    except Exception as e:
        print(f"   ⚠ Could not create indexes: {e}")
        pg_session.rollback()
    
    # Verify migration
    print("\n6. Verifying migration...")
    try:
        face_count = pg_session.execute(text("SELECT COUNT(*) FROM gallery")).scalar()
        voice_count = pg_session.execute(text("SELECT COUNT(*) FROM voice_gallery")).scalar()
        print(f"   ✓ PostgreSQL gallery has {face_count} face embeddings")
        print(f"   ✓ PostgreSQL gallery has {voice_count} voice embeddings")
    except Exception as e:
        print(f"   ⚠ Verification error: {e}")
    
    # Cleanup
    sqlite_session.close()
    pg_session.close()
    
    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("  1. Restart the backend: docker compose restart backend")
    print("  2. Test identification to verify pgvector is working")


if __name__ == "__main__":
    migrate_embeddings()
