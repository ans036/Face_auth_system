"""
Tests for the health check endpoint.
"""

import pytest


class TestHealthEndpoint:
    """Test cases for the /health endpoint."""
    
    def test_health_endpoint_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Health response should contain expected fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "service" in data
        assert data["service"] == "face-auth-backend"
        assert "version" in data
        assert "features" in data
        assert isinstance(data["features"], list)
    
    def test_health_version_is_valid(self, client):
        """Health version should be a valid semver string."""
        response = client.get("/health")
        data = response.json()
        
        version = data["version"]
        # Should be in format X.Y.Z
        parts = version.split(".")
        assert len(parts) >= 2
        assert all(part.isdigit() for part in parts)
    
    def test_health_features_include_core_modules(self, client):
        """Health features should include core biometric modules."""
        response = client.get("/health")
        data = response.json()
        
        features = data["features"]
        # Core features expected
        assert "face" in features
        assert "voice" in features
        assert "liveness" in features
