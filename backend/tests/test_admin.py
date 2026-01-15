"""
Tests for the admin authentication and management API.
"""

import pytest


class TestAdminLogin:
    """Test cases for admin login functionality."""
    
    def test_admin_login_success(self, client):
        """Valid admin credentials should return success and set session cookie."""
        response = client.post(
            "/admin/login",
            data={"username": "anish", "password": "floyd003"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "admin_session" in response.cookies
    
    def test_admin_login_wrong_password(self, client):
        """Wrong password should return 401 with appropriate message."""
        response = client.post(
            "/admin/login",
            data={"username": "anish", "password": "wrongpassword"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "Invalid password" in data["detail"]
    
    def test_admin_login_wrong_username(self, client):
        """Non-admin username should return 401 with access denied."""
        response = client.post(
            "/admin/login",
            data={"username": "notadmin", "password": "somepassword"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "not an Admin" in data["detail"]
    
    def test_admin_login_missing_fields(self, client):
        """Missing username or password should return 422."""
        response = client.post("/admin/login", data={})
        assert response.status_code == 422


class TestAdminAuthentication:
    """Test cases for admin authentication status."""
    
    def test_admin_check_unauthenticated(self, client):
        """Check endpoint should return false when not authenticated."""
        response = client.get("/admin/check")
        
        assert response.status_code == 200
        data = response.json()
        assert data["authenticated"] is False
    
    def test_admin_check_authenticated(self, client):
        """Check endpoint should return true after successful login."""
        # Login first
        client.post(
            "/admin/login",
            data={"username": "anish", "password": "floyd003"}
        )
        
        # Check authentication
        response = client.get("/admin/check")
        data = response.json()
        assert data["authenticated"] is True


class TestAdminProtectedEndpoints:
    """Test cases for admin-protected endpoints."""
    
    def test_admin_stats_requires_auth(self, client):
        """Stats endpoint should return 401 without authentication."""
        response = client.get("/admin/stats")
        assert response.status_code == 401
    
    def test_admin_logs_requires_auth(self, client):
        """Logs endpoint should return 401 without authentication."""
        response = client.get("/admin/logs")
        assert response.status_code == 401
    
    def test_admin_attempts_requires_auth(self, client):
        """Attempts endpoint should return 401 without authentication."""
        response = client.get("/admin/attempts")
        assert response.status_code == 401
    
    def test_admin_unauthorized_images_requires_auth(self, client):
        """Unauthorized images endpoint should return 401 without authentication."""
        response = client.get("/admin/unauthorized-images")
        assert response.status_code == 401
    
    def test_admin_stats_with_auth(self, client):
        """Stats endpoint should return data when authenticated."""
        # Login first
        client.post(
            "/admin/login",
            data={"username": "anish", "password": "floyd003"}
        )
        
        response = client.get("/admin/stats")
        assert response.status_code == 200
        data = response.json()
        
        # Verify expected stat fields exist
        assert "total_users" in data
        assert "face_embeddings" in data
        assert "voice_embeddings" in data
    
    def test_admin_logs_with_auth(self, client):
        """Logs endpoint should return data when authenticated."""
        # Login first
        client.post(
            "/admin/login",
            data={"username": "anish", "password": "floyd003"}
        )
        
        response = client.get("/admin/logs")
        assert response.status_code == 200
        data = response.json()
        
        assert "logs" in data
        assert "total" in data


class TestAdminLogout:
    """Test cases for admin logout."""
    
    def test_admin_logout(self, client):
        """Logout should invalidate session."""
        # Login first
        client.post(
            "/admin/login",
            data={"username": "anish", "password": "floyd003"}
        )
        
        # Verify logged in
        check_response = client.get("/admin/check")
        assert check_response.json()["authenticated"] is True
        
        # Logout
        client.get("/admin/logout", follow_redirects=False)
        
        # Clear cookies manually since logout redirects
        client.cookies.clear()
        
        # Verify logged out
        check_response = client.get("/admin/check")
        assert check_response.json()["authenticated"] is False
