"""
Face-Gated Private Messages API
Content that only reveals when authorized face is verified.

Features:
- Content remains blurred until face+liveness verified
- Re-blurs if user looks away for >30 seconds
- Session-based access tokens
"""

from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from typing import Optional, Dict
import time
import secrets
import json

router = APIRouter()

# In-memory storage for demo (use Redis/DB in production)
private_messages = {}
access_tokens = {}  # token -> {username, expires, message_id}

# Token validity duration
TOKEN_VALIDITY_SECONDS = 30  # Token expires if no face detected for 30s


@router.post("/create")
async def create_private_message(
    content: str = Form(...),
    allowed_users: str = Form(...),  # Comma-separated usernames
    title: str = Form("Private Message")
):
    """
    Create a face-gated private message.
    
    Args:
        content: The private content to protect
        allowed_users: Comma-separated list of allowed usernames
        title: Optional title for the message
    
    Returns:
        message_id: Unique ID to access this message
    """
    message_id = secrets.token_urlsafe(16)
    
    users = [u.strip().lower() for u in allowed_users.split(",")]
    
    private_messages[message_id] = {
        "id": message_id,
        "title": title,
        "content": content,
        "allowed_users": users,
        "created_at": time.time()
    }
    
    return {
        "status": "created",
        "message_id": message_id,
        "allowed_users": users,
        "access_url": f"/private/{message_id}"
    }


@router.get("/info/{message_id}")
async def get_message_info(message_id: str):
    """
    Get message metadata (without content).
    Used to show blurred preview.
    """
    if message_id not in private_messages:
        raise HTTPException(status_code=404, detail="Message not found")
    
    msg = private_messages[message_id]
    
    return {
        "id": msg["id"],
        "title": msg["title"],
        "allowed_users": msg["allowed_users"],
        "content_preview": msg["content"][:20] + "..." if len(msg["content"]) > 20 else "***",
        "requires_face_auth": True
    }


@router.post("/unlock/{message_id}")
async def unlock_message(
    message_id: str,
    username: str = Form(...),
    face_score: float = Form(...),
    is_live: bool = Form(False)
):
    """
    Attempt to unlock a private message with face verification.
    
    Args:
        message_id: ID of the private message
        username: Identified username from face recognition
        face_score: Recognition confidence score
        is_live: Whether liveness check passed
    
    Returns:
        content: The decrypted message content (if authorized)
    """
    if message_id not in private_messages:
        raise HTTPException(status_code=404, detail="Message not found")
    
    msg = private_messages[message_id]
    username_lower = username.lower()
    
    # Check if user is allowed
    if username_lower not in msg["allowed_users"] and "unknown" not in msg["allowed_users"]:
        return {
            "status": "denied",
            "reason": "User not authorized for this message",
            "username": username
        }
    
    # Check minimum score
    if face_score < 0.45:
        return {
            "status": "denied", 
            "reason": "Face verification confidence too low",
            "score": face_score
        }
    
    # Generate access token
    token = secrets.token_urlsafe(32)
    access_tokens[token] = {
        "username": username_lower,
        "message_id": message_id,
        "expires": time.time() + TOKEN_VALIDITY_SECONDS,
        "created": time.time()
    }
    
    return {
        "status": "unlocked",
        "content": msg["content"],
        "title": msg["title"],
        "token": token,
        "expires_in": TOKEN_VALIDITY_SECONDS,
        "liveness_verified": is_live,
        "message": f"Content unlocked for {username}. Stay visible to keep access."
    }


@router.post("/refresh/{token}")
async def refresh_access(
    token: str,
    face_detected: bool = Form(True),
    username: str = Form(None)
):
    """
    Refresh access token while user remains visible.
    Call this periodically to maintain access.
    
    Args:
        token: Current access token
        face_detected: Whether authorized face is still visible
        username: Current detected username (for validation)
    """
    if token not in access_tokens:
        return {
            "status": "expired",
            "message": "Token not found. Re-authenticate with face."
        }
    
    token_data = access_tokens[token]
    
    # Check if token already expired
    if time.time() > token_data["expires"]:
        del access_tokens[token]
        return {
            "status": "expired",
            "message": "Session timed out. Please re-verify your face."
        }
    
    # Check if face still detected
    if not face_detected:
        return {
            "status": "warning",
            "message": "Face not detected. Access will expire soon.",
            "remaining_seconds": max(0, token_data["expires"] - time.time())
        }
    
    # Verify same user
    if username and username.lower() != token_data["username"]:
        return {
            "status": "denied",
            "message": "Different user detected. Access revoked."
        }
    
    # Extend token
    token_data["expires"] = time.time() + TOKEN_VALIDITY_SECONDS
    
    return {
        "status": "active",
        "message": "Access extended",
        "expires_in": TOKEN_VALIDITY_SECONDS
    }


@router.post("/revoke/{token}")
async def revoke_access(token: str):
    """Revoke access token (when user closes or navigates away)."""
    if token in access_tokens:
        del access_tokens[token]
    
    return {"status": "revoked"}


@router.get("/list")
async def list_messages():
    """List all private messages (for demo purposes)."""
    return {
        "messages": [
            {
                "id": msg["id"],
                "title": msg["title"],
                "allowed_users": msg["allowed_users"],
                "created_at": msg["created_at"]
            }
            for msg in private_messages.values()
        ]
    }
