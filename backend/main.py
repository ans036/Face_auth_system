from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.enroll import router as enroll_router
from api.authenticate import router as auth_router
from api.identify import router as identify_router
from api.security import router as security_router
from api.gallery import router as gallery_router
from db.session import init_db
from scripts.build_gallery import build_gallery_on_startup

app = FastAPI(
    title="Face Authentication System",
    description="Face recognition and authentication API with InsightFace buffalo_l model",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "version": "2.0.0"
    }


@app.on_event("startup")
def startup():
    print("🚀 Starting Face Authentication System v2.0...")
    
    # Initialize database
    print("📊 Initializing database...")
    init_db()
    
    # Ensure directories exist
    import os
    os.makedirs("/app/unauthorized_attempts", exist_ok=True)
    
    # Build gallery from database folder
    print("🖼️  Building gallery...")
    build_gallery_on_startup()
    
    # Load gallery into recognizer
    print("🔄 Loading gallery into recognizer...")
    from api.identify import recognizer
    recognizer.load_gallery()
    
    print("✅ System ready!")


# Mount routers
app.include_router(identify_router, prefix="/identify", tags=["Identification"])
app.include_router(enroll_router, prefix="/enroll", tags=["Enrollment"])
app.include_router(gallery_router, prefix="/gallery", tags=["Gallery Management"])
app.include_router(security_router, prefix="/security", tags=["Security"])