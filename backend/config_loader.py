import yaml
import os

def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    conf_dir = os.path.join(base, "config")
    cfg = {}
    thresholds = load_yaml(os.path.join(conf_dir, "thresholds.yaml"))
    security = load_yaml(os.path.join(conf_dir, "security.yaml"))
    cfg.update(thresholds or {})
    cfg.update({"logging": security.get("logging", {})} if security else {})
    cfg.update(security or {})
    return cfg
