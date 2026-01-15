"""
Gallery Management API for listing, deleting, and rebuilding face galleries.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from db.crud import get_gallery_stats, delete_user_from_gallery, clear_gallery
from scripts.build_gallery import build_gallery

router = APIRouter()


@router.get("/")
async def list_gallery():
    """
    List all enrolled users and their embedding counts.
    
    Returns:
        Dictionary with user names and counts
    """
    try:
        stats = get_gallery_stats()
        return {
            "status": "success",
            "total_users": len(stats),
            "total_embeddings": sum(stats.values()),
            "users": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{username}")
async def get_user_gallery(username: str):
    """
    Get enrollment info for a specific user.
    """
    try:
        stats = get_gallery_stats()
        if username not in stats:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found in gallery")
        
        return {
            "status": "success",
            "username": username,
            "embedding_count": stats[username]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{username}")
async def delete_user(username: str):
    """
    Remove a user from the gallery.
    
    This deletes all embeddings for the specified user.
    """
    try:
        stats = get_gallery_stats()
        if username not in stats:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found in gallery")
        
        count = stats[username]
        delete_user_from_gallery(username)
        
        # Reload recognizer gallery
        try:
            from api.identify import recognizer
            recognizer.reload_gallery()
        except:
            pass
        
        return {
            "status": "success",
            "message": f"Deleted {count} embeddings for user '{username}'",
            "username": username,
            "deleted_count": count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebuild")
async def rebuild_gallery(database_path: Optional[str] = None):
    """
    Rebuild the gallery from the database folder.
    
    This will:
    1. Clear the current gallery
    2. Re-scan the database folder for images
    3. Generate new embeddings
    4. Reload the recognizer
    
    Args:
        database_path: Optional custom path (defaults to /app/database)
    """
    try:
        path = database_path or "/app/database"
        
        print(f"🔄 Rebuilding gallery from {path}...")
        build_gallery(path, min_quality=0.20)
        
        # Reload recognizer
        try:
            from api.identify import recognizer
            recognizer.reload_gallery()
        except:
            pass
        
        # Get new stats
        stats = get_gallery_stats()
        
        return {
            "status": "success",
            "message": "Gallery rebuilt successfully",
            "total_users": len(stats),
            "total_embeddings": sum(stats.values()),
            "users": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/")
async def clear_all_gallery():
    """
    Clear all embeddings from the gallery.
    
    ⚠️ WARNING: This deletes ALL enrolled faces!
    """
    try:
        stats = get_gallery_stats()
        total = sum(stats.values())
        
        clear_gallery()
        
        # Reload recognizer
        try:
            from api.identify import recognizer
            recognizer.reload_gallery()
        except:
            pass
        
        return {
            "status": "success",
            "message": f"Cleared {total} embeddings from gallery",
            "deleted_count": total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
