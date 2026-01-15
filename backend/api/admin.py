"""
Admin Panel API
Provides authentication and management endpoints for administrators.
"""

from fastapi import APIRouter, HTTPException, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import hashlib
import secrets
import os
import time
from sqlalchemy.orm import Session

router = APIRouter(prefix="/admin", tags=["admin"])

# Simple session store (in production, use Redis or database)
SESSIONS = {}

# Admin credentials from environment or defaults
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "anish")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "floyd003")
SECRET_KEY = os.getenv("SECRET_KEY", "face-auth-secret-key-change-in-production")


# ============ LOG CACHE (OPTIMIZATION) ============
class LogCache:
    """Cache parsed log entries to avoid re-reading file on every request."""
    
    def __init__(self, ttl_seconds: int = 30):
        self.ttl = ttl_seconds
        self._cache = None
        self._cache_time = 0
        self._file_mtime = 0
    
    def get_parsed_logs(self, log_file: str) -> List[dict]:
        """Get parsed log entries, using cache if valid."""
        current_time = time.time()
        
        # Check if file was modified
        try:
            file_mtime = os.path.getmtime(log_file) if os.path.exists(log_file) else 0
        except:
            file_mtime = 0
        
        # Return cached data if still valid
        if (self._cache is not None and 
            current_time - self._cache_time < self.ttl and 
            file_mtime == self._file_mtime):
            return self._cache
        
        # Re-parse the log file
        parsed = []
        try:
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            entry = parse_log_line(line)
                            if entry:
                                parsed.append(entry)
        except Exception as e:
            print(f"⚠️ Error parsing log cache: {e}")
        
        # Update cache
        self._cache = parsed
        self._cache_time = current_time
        self._file_mtime = file_mtime
        
        return parsed


# Global log cache instance
_log_cache = LogCache(ttl_seconds=30)

def hash_password(password: str) -> str:
    """Hash password with salt."""
    return hashlib.sha256((password + SECRET_KEY).encode()).hexdigest()

def verify_session(request: Request) -> bool:
    """Verify if request has valid admin session."""
    session_id = request.cookies.get("admin_session")
    if not session_id:
        return False
    session = SESSIONS.get(session_id)
    if not session:
        return False
    if datetime.now() > session["expires"]:
        del SESSIONS[session_id]
        return False
    return True

def require_admin(request: Request):
    """Dependency to require admin authentication."""
    if not verify_session(request):
        raise HTTPException(status_code=401, detail="Admin authentication required")
    return True


class LoginRequest(BaseModel):
    username: str
    password: str


class AttemptConfirmation(BaseModel):
    status: str  # 'confirmed' or 'rejected'
    notes: Optional[str] = None


@router.post("/login")
async def admin_login(response: Response, username: str = Form(...), password: str = Form(...)):
    """Admin login endpoint."""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        # Create session
        session_id = secrets.token_urlsafe(32)
        SESSIONS[session_id] = {
            "username": username,
            "created": datetime.now(),
            "expires": datetime.now() + timedelta(hours=8)
        }
        
        # Set cookie and return success
        response = JSONResponse(content={"status": "success", "message": "Login successful"})
        response.set_cookie(
            key="admin_session",
            value=session_id,
            httponly=True,
            max_age=8*60*60  # 8 hours
        )
        return response
    
    # Custom error message for non-admin or wrong password
    msg = "Access denied - not an Admin" if username != ADMIN_USERNAME else "Invalid password"
    raise HTTPException(status_code=401, detail=msg)


@router.get("/logout")
async def admin_logout(request: Request, response: Response):
    """Admin logout endpoint."""
    session_id = request.cookies.get("admin_session")
    if session_id and session_id in SESSIONS:
        del SESSIONS[session_id]
    
    response = RedirectResponse(url="/admin", status_code=303)
    response.delete_cookie("admin_session")
    return response


@router.get("/check")
async def check_auth(request: Request):
    """Check if admin is authenticated."""
    return {"authenticated": verify_session(request)}


@router.get("/logs")
async def get_security_logs(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    _: bool = Depends(require_admin)
):
    """Get security log entries."""
    logs = []
    log_file = "/app/security.log"
    
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                lines = f.readlines()
                # Parse log lines (reverse for newest first)
                for line in reversed(lines[-limit-offset:offset if offset else None]):
                    line = line.strip()
                    if line:
                        logs.append({"raw": line})
    except Exception as e:
        print(f"Error reading logs: {e}")
    
    return {"logs": logs[:limit], "total": len(logs)}


@router.get("/attempts")
async def get_auth_attempts(
    request: Request,
    status: Optional[str] = None,  # 'authorized', 'unauthorized', 'all'
    limit: int = 50,
    _: bool = Depends(require_admin)
):
    """Get authentication attempts from logs (uses cache for performance)."""
    log_file = "/app/security.log"
    
    # Use cached parsed logs
    all_entries = _log_cache.get_parsed_logs(log_file)
    
    # Filter by status
    attempts = []
    for entry in all_entries:
        if status == "authorized" and entry.get("is_known"):
            attempts.append(entry)
        elif status == "unauthorized" and not entry.get("is_known"):
            attempts.append(entry)
        elif status == "all" or status is None:
            attempts.append(entry)
    
    # Return newest first
    return {"attempts": list(reversed(attempts[-limit:]))}


def parse_log_line(line: str) -> Optional[dict]:
    """Parse a security log line into structured data."""
    try:
        # Expected format: [2026-01-15 10:30:45] | User: Anish | Score: 0.45 | Live: True | Manual: False
        parts = line.split(" | ")
        if len(parts) < 4:
            return None
        
        timestamp_str = parts[0].strip("[]")
        user_part = parts[1] if len(parts) > 1 else ""
        score_part = parts[2] if len(parts) > 2 else ""
        live_part = parts[3] if len(parts) > 3 else ""
        
        username = user_part.replace("User:", "").strip() if "User:" in user_part else "Unknown"
        score = 0.0
        if "Score:" in score_part:
            try:
                score = float(score_part.replace("Score:", "").strip())
            except:
                pass
        
        is_live = "True" in live_part
        is_known = username != "Unknown"
        
        return {
            "timestamp": timestamp_str,
            "username": username,
            "score": score,
            "is_live": is_live,
            "is_known": is_known,
            "raw": line
        }
    except Exception as e:
        return {"raw": line, "error": str(e)}


@router.get("/stats")
async def get_dashboard_stats(
    request: Request,
    _: bool = Depends(require_admin)
):
    """Get dashboard statistics."""
    from db.crud import get_all_gallery, get_voice_gallery
    import re
    
    stats = {
        "total_users": 0,
        "face_embeddings": 0,
        "voice_embeddings": 0,
        "total_attempts": 0,
        "authorized_today": 0,
        "unauthorized_today": 0
    }
    
    try:
        face_gallery = get_all_gallery()
        voice_gallery = get_voice_gallery()
        
        stats["face_embeddings"] = len(face_gallery)
        stats["voice_embeddings"] = len(voice_gallery)
        
        # Count unique users
        users = set()
        for entry in face_gallery:
            users.add(entry.get("username", ""))
        stats["total_users"] = len(users)
        
        # Count today's attempts from log
        log_file = "/app/security.log"
        today = datetime.now().strftime("%Y-%m-%d")
        # Get today's midnight timestamp for comparison
        today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f.readlines():
                    is_today = False
                    is_unauthorized = False
                    
                    # Check old format: "2026-01-15 ..."
                    if today in line:
                        is_today = True
                    
                    # Check new format: "[1768491027] ..."
                    ts_match = re.match(r'^\[(\d+)\]', line)
                    if ts_match:
                        ts = int(ts_match.group(1))
                        if ts >= today_midnight:
                            is_today = True
                    
                    if is_today:
                        stats["total_attempts"] += 1
                        # Check for unauthorized (Unknown or null username)
                        if "Unknown" in line or '"username": null' in line:
                            stats["unauthorized_today"] += 1
                        else:
                            stats["authorized_today"] += 1
    except Exception as e:
        print(f"Error getting stats: {e}")
    
    return stats


@router.get("/unauthorized-images")
async def get_unauthorized_images(
    request: Request,
    limit: int = 20,
    _: bool = Depends(require_admin)
):
    """Get list of unauthorized attempt images."""
    images = []
    image_dir = "/app/unauthorized_attempts"
    
    try:
        if os.path.exists(image_dir):
            files = sorted(os.listdir(image_dir), reverse=True)[:limit]
            for f in files:
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    images.append({
                        "filename": f,
                        "path": f"/unauthorized_attempts/{f}",
                        "timestamp": f.split("_")[0] if "_" in f else f
                    })
    except Exception as e:
        print(f"Error listing images: {e}")
    
    return {"images": images}
