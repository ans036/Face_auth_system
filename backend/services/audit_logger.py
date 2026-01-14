import logging
import json
import time
from config_loader import load_config

cfg = load_config()
LOG_FILE = cfg.get("logging", {}).get("security_log", "security.log")

# basic logger
logger = logging.getLogger("security")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)

def log_security_event(username, success: bool, score: float, reason: str):
    entry = {
        "timestamp": time.time(),
        "username": username,
        "success": bool(success),
        "score": float(score),
        "reason": reason
    }
    logger.info(json.dumps(entry))
    # also print to stdout for convenience
    print("SECURITY:", entry)
