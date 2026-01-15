from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.enroll import router as enroll_router
from api.authenticate import router as auth_router
from api.identify import router as identify_router
from api.security import router as security_router
from api.gallery import router as gallery_router
from api.private_message import router as private_router
from api.admin import router as admin_router
from db.session import init_db
from scripts.build_gallery import build_gallery_on_startup

app = FastAPI(
    title="Multi-Modal Face Authentication System",
    description="Face + voice recognition with liveness detection and face-gated messaging",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "face-auth-backend",
        "version": "3.1.0",
        "features": ["face", "voice", "liveness", "private_messages"]
    }


@app.on_event("startup")
def startup():
    print("🚀 Starting Multi-Modal Face Auth System v3.0...")
    
    # Initialize database
    print("📊 Initializing database...")
    init_db()
    
    # Ensure directories exist
    import os
    os.makedirs("/app/unauthorized_attempts", exist_ok=True)
    
    # Build gallery from database folder
    print("🖼️  Building face gallery...")
    build_gallery_on_startup()
    
    # Load galleries (Face + Voice)
    print("🔄 Loading galleries into multimodal authenticator...")
    try:
        from api.identify import multimodal_auth
        multimodal_auth.load_galleries()
    except Exception as e:
        print(f"⚠️ Failed to load galleries: {e}")
    
    print("✅ System ready! Features: Face + Voice + Liveness + Private Messages")


# Mount routers
app.include_router(identify_router, prefix="/identify", tags=["Identification"])
app.include_router(enroll_router, prefix="/enroll", tags=["Enrollment"])
app.include_router(gallery_router, prefix="/gallery", tags=["Gallery Management"])
app.include_router(security_router, prefix="/security", tags=["Security"])
app.include_router(private_router, prefix="/private", tags=["Private Messages"])
app.include_router(admin_router, tags=["Admin"])