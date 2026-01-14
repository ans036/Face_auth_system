from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, LargeBinary

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

class Gallery(Base):
    __tablename__ = "gallery"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
