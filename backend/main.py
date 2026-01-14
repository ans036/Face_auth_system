from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.enroll import router as enroll_router
from api.authenticate import router as auth_router
from api.identify import router as identify_router
from api.health import router as health_router
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
    # create DB tables
    init_db()
    # build gallery embeddings from images stored under /app/database
    build_gallery_on_startup()

app.include_router(enroll_router, prefix="/enroll", tags=["enroll"])
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(identify_router, prefix="/identify", tags=["identify"])
app.include_router(health_router, prefix="/health", tags=["health"])
