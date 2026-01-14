from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.enroll import router as enroll_router
from api.authenticate import router as auth_router
from api.identify import router as identify_router
from api.health import router as health_router
from api.security import router as security_router
from db.session import init_db
from scripts.build_gallery import build_gallery_on_startup

app = FastAPI(title="Face Authentication System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows the frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def startup():
    init_db()
    # Ensure the snapshot directory exists on startup
    import os
    os.makedirs("/app/unauthorized_attempts", exist_ok=True)
    build_gallery_on_startup()

app.include_router(identify_router, prefix="/identify")
# Register the new router prefix
app.include_router(security_router, prefix="/security")
