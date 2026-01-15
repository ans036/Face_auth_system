from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, LargeBinary, Float

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

class Gallery(Base):
    """Face embeddings gallery."""
    __tablename__ = "gallery"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

class VoiceGallery(Base):
    """Voice embeddings gallery for multi-modal auth."""
    __tablename__ = "voice_gallery"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
