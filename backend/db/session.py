"""
Database session management with PostgreSQL + pgvector support.

Supports both PostgreSQL (production) and SQLite (development/testing).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base, create_vector_indexes, USE_PGVECTOR
import os

# Get database URL from environment (PostgreSQL) or use SQLite fallback
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/face_auth.db")

# Handle SQLite special case
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL - use connection pool
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True  # Verify connections before use
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



def init_db():
    """Initialize database tables and indexes."""
    print(f"📊 Initializing database: {'PostgreSQL+pgvector' if USE_PGVECTOR else 'SQLite'}")
    
    # Enable pgvector extension BEFORE creating tables
    if USE_PGVECTOR:
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            print("✅ pgvector extension enabled")
        except Exception as e:
            print(f"⚠️ Could not enable pgvector extension: {e}")

    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create vector indexes for PostgreSQL
    if USE_PGVECTOR:
        try:
            create_vector_indexes(engine)
        except Exception as e:
            print(f"⚠️ Could not create vector indexes: {e}")
    
    print("✅ Database initialized successfully")


def get_db():
    """Dependency for FastAPI - yields database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()