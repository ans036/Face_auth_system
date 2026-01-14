from fastapi import APIRouter
from services.audit_logger import log_security_event

router = APIRouter()

@router.get("/")
def health():
    return {"status": "ok"}
