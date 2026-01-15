"""
Tests for the enrollment API endpoints.
"""

import pytest
import io


class TestEnrollValidation:
    """Test cases for enrollment input validation."""
    
    def test_enroll_empty_username(self, client, sample_image_bytes):
        """Empty username should return 400 or 422 validation error."""
        response = client.post(
            "/enroll/",
            data={"username": ""},
            files={"files": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        )
        
        # Empty username is caught by validation (400 or 422)
        assert response.status_code in [400, 422]
    
    def test_enroll_whitespace_username(self, client, sample_image_bytes):
        """Whitespace-only username should return 400 error."""
        response = client.post(
            "/enroll/",
            data={"username": "   "},
            files={"files": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        )
        
        assert response.status_code == 400
    
    def test_enroll_missing_files(self, client):
        """Missing files should return 422 validation error."""
        response = client.post(
            "/enroll/",
            data={"username": "testuser"}
        )
        
        assert response.status_code == 422


class TestEnrollInstructions:
    """Test cases for enrollment instructions endpoint."""
    
    def test_enroll_instructions_returns_200(self, client):
        """Save-from-camera instructions endpoint should return 200."""
        response = client.get("/enroll/save-from-camera")
        assert response.status_code == 200
    
    def test_enroll_instructions_structure(self, client):
        """Instructions should contain expected fields."""
        response = client.get("/enroll/save-from-camera")
        data = response.json()
        
        assert "instructions" in data
        assert isinstance(data["instructions"], list)
        assert len(data["instructions"]) > 0
        assert "endpoint" in data
        assert "required_fields" in data


class TestSingleEnrollEndpoint:
    """Test cases for the legacy single-image enrollment endpoint."""
    
    def test_single_enroll_endpoint_exists(self, client, sample_image_bytes):
        """Single enroll endpoint should exist and accept request."""
        response = client.post(
            "/enroll/single",
            data={"username": "testuser"},
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        )
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
