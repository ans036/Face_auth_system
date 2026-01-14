import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
SNAPSHOT_DIR = "/app/unauthorized_attempts"

@router.get("/snapshots/")
def list_snapshots():
    """Lists all saved unauthorized snapshot filenames, newest first."""
    if not os.path.exists(SNAPSHOT_DIR):
        return []
    
    # Get list of .jpg files
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".jpg")]
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x)), reverse=True)
    return files

@router.get("/snapshots/{filename}")
def get_snapshot(filename: str):
    """Serves a specific snapshot image."""
    # Basic security against path traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    file_path = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Snapshot not found")
        
    return FileResponse(file_path, media_type="image/jpeg")