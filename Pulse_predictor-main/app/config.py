import os
import secrets

_default_key = secrets.token_hex(32)
SECRET_KEY = os.getenv("PPP_SECRET_KEY", _default_key)
if "PPP_SECRET_KEY" not in os.environ:
    import warnings
    warnings.warn("PPP_SECRET_KEY not set! Using random key — sessions won't persist across restarts.", stacklevel=1)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

DATABASE_URL = "sqlite:///./ppp.db"

# Alert thresholds (configurable via environment variables)
COST_VARIANCE_THRESHOLD = float(os.getenv("COST_VARIANCE_THRESHOLD", "0.15"))
EFFORT_VARIANCE_THRESHOLD = float(os.getenv("EFFORT_VARIANCE_THRESHOLD", "0.15"))
RISK_SCORE_THRESHOLD = float(os.getenv("RISK_SCORE_THRESHOLD", "0.8"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
