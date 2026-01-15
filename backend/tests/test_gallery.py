"""
Tests for gallery CRUD operations and API endpoints.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestGalleryCRUD:
    """Test cases for gallery database operations."""
    
    def test_create_gallery_entry(self, test_session):
        """Should create a gallery entry successfully."""
        from db.models import Gallery
        
        username = "testuser"
        embedding = np.random.randn(512).astype(np.float32)
        
        entry = Gallery(username=username, embedding=embedding.tobytes())
        test_session.add(entry)
        test_session.commit()
        
        # Verify entry was created
        result = test_session.query(Gallery).filter_by(username=username).first()
        assert result is not None
        assert result.username == username
    
    def test_get_all_gallery_empty(self, test_session):
        """Empty gallery should return empty list."""
        from db.models import Gallery
        
        results = test_session.query(Gallery).all()
        assert len(results) == 0
    
    def test_get_all_gallery_with_entries(self, test_session):
        """Gallery with entries should return all entries."""
        from db.models import Gallery
        
        # Add test entries
        for i in range(3):
            embedding = np.random.randn(512).astype(np.float32)
            entry = Gallery(username=f"user{i}", embedding=embedding.tobytes())
            test_session.add(entry)
        test_session.commit()
        
        results = test_session.query(Gallery).all()
        assert len(results) == 3
    
    def test_delete_user_from_gallery(self, test_session):
        """Should delete all entries for a specific user."""
        from db.models import Gallery
        
        # Add entries for two users
        for i in range(2):
            embedding = np.random.randn(512).astype(np.float32)
            test_session.add(Gallery(username="user_to_delete", embedding=embedding.tobytes()))
        embedding = np.random.randn(512).astype(np.float32)
        test_session.add(Gallery(username="user_to_keep", embedding=embedding.tobytes()))
        test_session.commit()
        
        # Delete one user
        test_session.query(Gallery).filter_by(username="user_to_delete").delete()
        test_session.commit()
        
        # Verify only the kept user remains
        results = test_session.query(Gallery).all()
        assert len(results) == 1
        assert results[0].username == "user_to_keep"
    
    def test_gallery_stats(self, test_session):
        """Should count embeddings per user correctly."""
        from db.models import Gallery
        
        # Add entries: user1 has 3, user2 has 2
        for i in range(3):
            embedding = np.random.randn(512).astype(np.float32)
            test_session.add(Gallery(username="user1", embedding=embedding.tobytes()))
        for i in range(2):
            embedding = np.random.randn(512).astype(np.float32)
            test_session.add(Gallery(username="user2", embedding=embedding.tobytes()))
        test_session.commit()
        
        # Count per user
        from sqlalchemy import func
        stats = test_session.query(
            Gallery.username,
            func.count(Gallery.id)
        ).group_by(Gallery.username).all()
        
        stats_dict = {username: count for username, count in stats}
        assert stats_dict.get("user1") == 3
        assert stats_dict.get("user2") == 2


class TestVoiceGalleryCRUD:
    """Test cases for voice gallery database operations."""
    
    def test_create_voice_entry(self, test_session):
        """Should create a voice gallery entry successfully."""
        from db.models import VoiceGallery
        
        username = "testuser"
        embedding = np.random.randn(192).astype(np.float32)  # ECAPA-TDNN dimension
        
        entry = VoiceGallery(username=username, embedding=embedding.tobytes())
        test_session.add(entry)
        test_session.commit()
        
        result = test_session.query(VoiceGallery).filter_by(username=username).first()
        assert result is not None
        assert result.username == username
    
    def test_get_voice_gallery(self, test_session):
        """Should retrieve all voice entries."""
        from db.models import VoiceGallery
        
        # Add test entries
        for i in range(2):
            embedding = np.random.randn(192).astype(np.float32)
            entry = VoiceGallery(username=f"voice_user{i}", embedding=embedding.tobytes())
            test_session.add(entry)
        test_session.commit()
        
        results = test_session.query(VoiceGallery).all()
        assert len(results) == 2


class TestGalleryAPI:
    """Test cases for gallery API endpoints."""
    
    def test_gallery_rebuild_endpoint_exists(self, client):
        """Gallery rebuild endpoint should exist."""
        # Login as admin first (if required) - rebuild may be protected
        client.post(
            "/admin/login",
            data={"username": "anish", "password": "floyd003"}
        )
        
        response = client.post("/gallery/rebuild")
        # Should not be 404
        assert response.status_code != 404
