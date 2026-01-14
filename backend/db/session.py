from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base
import os

DATABASE_URL = "sqlite:////app/face_auth.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

# The missing function causing your error
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()