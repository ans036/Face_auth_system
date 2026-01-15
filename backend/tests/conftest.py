"""
Pytest configuration and fixtures for Face Auth System tests.
Provides test client, in-memory database, and mock fixtures for ML models.
"""

import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock
import numpy as np

# Import database components
from db.models import Base
from db.session import get_db


# Create in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def test_engine():
    """Create a fresh in-memory database engine for each test."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def mock_face_embedder():
    """Mock FaceEmbedder to avoid loading ML models."""
    mock = MagicMock()
    # Return a normalized 512-dim embedding vector
    mock.embed.return_value = np.random.randn(512).astype(np.float32)
    mock.embed.return_value /= np.linalg.norm(mock.embed.return_value)
    return mock


@pytest.fixture(scope="function")
def mock_face_detector():
    """Mock FaceDetector to avoid loading ML models."""
    mock = MagicMock()
    # Return a bounding box and 5 keypoints
    mock.detect.return_value = (
        [100, 100, 200, 200],  # box: x1, y1, x2, y2
        np.array([[120, 130], [180, 130], [150, 160], [125, 185], [175, 185]])  # 5 keypoints
    )
    return mock


@pytest.fixture(scope="function")
def mock_voice_embedder():
    """Mock VoiceEmbedder to avoid loading SpeechBrain models."""
    mock = MagicMock()
    # Return a normalized 192-dim embedding vector (ECAPA-TDNN output)
    mock.embed.return_value = np.random.randn(192).astype(np.float32)
    mock.embed.return_value /= np.linalg.norm(mock.embed.return_value)
    return mock


@pytest.fixture(scope="function")
def client(test_engine):
    """
    Create FastAPI TestClient with a minimal test app.
    We build our own app to avoid importing heavy ML dependencies.
    """
    # Create a test session factory bound to our test engine
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    # Import FastAPI components
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # Create a minimal test app
    test_app = FastAPI(
        title="Test App",
        version="3.1.0"
    )
    
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add health endpoint directly
    @test_app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "service": "face-auth-backend",
            "version": "3.1.0",
            "features": ["face", "voice", "liveness", "private_messages"]
        }
    
    # Import admin router directly (it has no ML dependencies)
    from api.admin import router as admin_router
    test_app.include_router(admin_router, tags=["Admin"])
    
    # Create mock enroll router (simpler than importing the real one)
    from fastapi import APIRouter, Form, UploadFile, File, HTTPException
    from typing import List
    
    enroll_router = APIRouter()
    
    @enroll_router.post("/")
    async def enroll_user(
        username: str = Form(...),
        files: List[UploadFile] = File(...)
    ):
        if not username or len(username.strip()) == 0:
            raise HTTPException(status_code=400, detail="Username cannot be empty")
        return {"status": "success", "username": username.strip()}
    
    @enroll_router.post("/single")
    async def enroll_single(
        username: str = Form(...),
        file: UploadFile = File(...)
    ):
        if not username or len(username.strip()) == 0:
            raise HTTPException(status_code=400, detail="Username cannot be empty")
        return {"status": "success", "username": username.strip()}
    
    @enroll_router.get("/save-from-camera")
    async def save_enrollment_instructions():
        return {
            "instructions": [
                "1. Use the frontend camera view to capture your face",
                "2. Save the captured images to the database folder"
            ],
            "endpoint": "POST /enroll/",
            "required_fields": {"username": "string", "files": "image files"}
        }
    
    test_app.include_router(enroll_router, prefix="/enroll", tags=["Enrollment"])
    
    # Create mock gallery router
    gallery_router = APIRouter()
    
    @gallery_router.post("/rebuild")
    async def rebuild_gallery():
        return {"status": "success", "message": "Gallery rebuilt"}
    
    test_app.include_router(gallery_router, prefix="/gallery", tags=["Gallery"])
    
    # Override database
    test_app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(test_app) as test_client:
        yield test_client
    
    test_app.dependency_overrides.clear()


@pytest.fixture
def sample_image_bytes():
    """Generate a simple test image as bytes."""
    import io
    from PIL import Image
    
    # Create a simple 200x200 RGB image
    img = Image.new('RGB', (200, 200), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def admin_session(client):
    """Create an authenticated admin session."""
    response = client.post(
        "/admin/login",
        data={"username": "anish", "password": "floyd003"}
    )
    # Return cookies for authenticated requests
    return client.cookies
