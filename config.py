"""
Centralized configuration for the Pork Weighing Analysis Web App.
All settings loaded from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from webapp directory first, then fall back to parent directory
_webapp_dir = Path(__file__).parent
_parent_dir = _webapp_dir.parent

# Load parent .env first (base config), then webapp .env (overrides)
load_dotenv(_parent_dir / ".env")
load_dotenv(_webapp_dir / ".env", override=True)


# =============================================================================
# API Keys (inherited from parent pipeline)
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# =============================================================================
# Agent-level defaults — each agent file owns its canonical parameter values.
# =============================================================================
import sys as _sys
_agents_dir = os.path.join(os.path.dirname(__file__), "agents")
if _agents_dir not in _sys.path:
    _sys.path.insert(0, _agents_dir)

# Maps agent name → module name inside agents/
_AGENT_MODULE_MAP = {
    "pork_weighing":  "pork_weighing_compliance",
    "plating_time":   "plating_time",
    "serve_time":          "avg_serve_time",
    "avg_serve_time":      "avg_serve_time",
    "noodle_rotation":     "noodle_rotation_compliance",
    "bowl_completion_rate": "bowl_completion_rate",
    "bowl_completion":     "bowl_completion_rate",
}

_AGENT_DEFAULTS_KEYS = [
    "AGENT_PHASE1_MODEL_NAME",
    "AGENT_PHASE2_MODEL_NAME",
    "AGENT_FPS",
    "AGENT_CONFIDENCE_THRESHOLD",
    "AGENT_MAX_BATCH_SIZE_MB",
    "AGENT_CLIP_BUFFER_SECONDS",
    "AGENT_MAX_FRAMES_PER_BATCH",
    "AGENT_BATCH_OVERLAP_FRAMES",
    "AGENT_IMAGE_QUALITY",
    "AGENT_IMAGE_UPSCALE_FACTOR",
    "AGENT_IMAGE_TARGET_RESOLUTION",
    "AGENT_IMAGE_FORMAT",
    "AGENT_PHASE2_IMAGE_FORMAT",
    "AGENT_IMAGE_INTERPOLATION",
    "AGENT_ENABLE_CROPPING",
    "AGENT_ROTATION_ANGLE",
]

_FALLBACK_AGENT_DEFAULTS = {
    "AGENT_PHASE1_MODEL_NAME":       "gpt-5-mini",
    "AGENT_PHASE2_MODEL_NAME":       "gemini-2.5-pro",
    "AGENT_FPS":                     1.0,
    "AGENT_CONFIDENCE_THRESHOLD":    0.7,
    "AGENT_MAX_BATCH_SIZE_MB":       30.0,
    "AGENT_CLIP_BUFFER_SECONDS":     2,
    "AGENT_MAX_FRAMES_PER_BATCH":    300,
    "AGENT_BATCH_OVERLAP_FRAMES":    2,
    "AGENT_IMAGE_QUALITY":           95,
    "AGENT_PHASE1_MAX_LONG_EDGE":    1024,
    "AGENT_IMAGE_UPSCALE_FACTOR":    1.0,
    "AGENT_IMAGE_TARGET_RESOLUTION": "auto",
    "AGENT_IMAGE_FORMAT":            "JPEG",
    "AGENT_PHASE2_IMAGE_FORMAT":     "PNG",
    "AGENT_IMAGE_INTERPOLATION":     "CUBIC",
    "AGENT_ENABLE_CROPPING":         True,
    "AGENT_ROTATION_ANGLE":          270,
}


def get_agent_defaults(agent_name: str) -> dict:
    """Return the AGENT_* defaults dict for the given agent.

    Imports from the agent's own module so each task owns its parameters.
    Falls back to _FALLBACK_AGENT_DEFAULTS if the module is unavailable.
    """
    module_name = _AGENT_MODULE_MAP.get(agent_name, "pork_weighing_compliance")
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return {k: getattr(mod, k) for k in _AGENT_DEFAULTS_KEYS if hasattr(mod, k)}
    except Exception:
        return dict(_FALLBACK_AGENT_DEFAULTS)


# Global defaults are sourced from pork_weighing (primary agent) for backward compat.
try:
    from pork_weighing_compliance import (
        AGENT_PHASE1_MODEL_NAME,
        AGENT_PHASE2_MODEL_NAME,
        AGENT_FPS,
        AGENT_CONFIDENCE_THRESHOLD,
        AGENT_MAX_BATCH_SIZE_MB,
        AGENT_CLIP_BUFFER_SECONDS,
        AGENT_MAX_FRAMES_PER_BATCH,
        AGENT_BATCH_OVERLAP_FRAMES,
        AGENT_IMAGE_QUALITY,
        AGENT_IMAGE_UPSCALE_FACTOR,
        AGENT_IMAGE_TARGET_RESOLUTION,
        AGENT_IMAGE_FORMAT,
        AGENT_PHASE2_IMAGE_FORMAT,
        AGENT_IMAGE_INTERPOLATION,
        AGENT_ENABLE_CROPPING,
        AGENT_ROTATION_ANGLE,
    )
except Exception:
    _fb = _FALLBACK_AGENT_DEFAULTS
    AGENT_PHASE1_MODEL_NAME       = _fb["AGENT_PHASE1_MODEL_NAME"]
    AGENT_PHASE2_MODEL_NAME       = _fb["AGENT_PHASE2_MODEL_NAME"]
    AGENT_FPS                     = _fb["AGENT_FPS"]
    AGENT_CONFIDENCE_THRESHOLD    = _fb["AGENT_CONFIDENCE_THRESHOLD"]
    AGENT_MAX_BATCH_SIZE_MB       = _fb["AGENT_MAX_BATCH_SIZE_MB"]
    AGENT_CLIP_BUFFER_SECONDS     = _fb["AGENT_CLIP_BUFFER_SECONDS"]
    AGENT_MAX_FRAMES_PER_BATCH    = _fb["AGENT_MAX_FRAMES_PER_BATCH"]
    AGENT_BATCH_OVERLAP_FRAMES    = _fb["AGENT_BATCH_OVERLAP_FRAMES"]
    AGENT_IMAGE_QUALITY           = _fb["AGENT_IMAGE_QUALITY"]
    AGENT_IMAGE_UPSCALE_FACTOR    = _fb["AGENT_IMAGE_UPSCALE_FACTOR"]
    AGENT_IMAGE_TARGET_RESOLUTION = _fb["AGENT_IMAGE_TARGET_RESOLUTION"]
    AGENT_IMAGE_FORMAT            = _fb["AGENT_IMAGE_FORMAT"]
    AGENT_PHASE2_IMAGE_FORMAT     = _fb["AGENT_PHASE2_IMAGE_FORMAT"]
    AGENT_IMAGE_INTERPOLATION     = _fb["AGENT_IMAGE_INTERPOLATION"]
    AGENT_ENABLE_CROPPING         = _fb["AGENT_ENABLE_CROPPING"]
    AGENT_ROTATION_ANGLE          = _fb["AGENT_ROTATION_ANGLE"]

# =============================================================================
# Model Configuration
# =============================================================================
PHASE1_MODEL_NAME = os.getenv("PHASE1_MODEL_NAME", AGENT_PHASE1_MODEL_NAME)
PHASE2_MODEL_NAME = os.getenv("PHASE2_MODEL_NAME", AGENT_PHASE2_MODEL_NAME)
AUTO_ROI_VLM_MODEL = os.getenv("AUTO_ROI_VLM_MODEL", "gemini-2.5-pro")
# Path to the root ROI knowledge-base folder.
# Structure: ROI_KB_DIR/{agent_name}/*.jpg  (plain or annotated reference frames)
# Leave unset to run without KB context.
ROI_KB_DIR = os.getenv("ROI_KB_DIR", os.path.join(os.path.dirname(__file__), "roi_kb"))
ENABLE_PHASE2 = os.getenv("ENABLE_PHASE2", "true").lower() == "true"

# =============================================================================
# Pipeline Settings — defaults sourced from the active agent file
# =============================================================================
FPS                  = float(os.getenv("FPS",                  str(AGENT_FPS)))
MAX_BATCH_SIZE_MB    = float(os.getenv("MAX_BATCH_SIZE_MB",    str(AGENT_MAX_BATCH_SIZE_MB)))
MAX_FRAMES_PER_BATCH = int(os.getenv("MAX_FRAMES_PER_BATCH",   str(AGENT_MAX_FRAMES_PER_BATCH)))
BATCH_OVERLAP_FRAMES = int(os.getenv("BATCH_OVERLAP_FRAMES",   str(AGENT_BATCH_OVERLAP_FRAMES)))
CLIP_BUFFER_SECONDS  = int(os.getenv("CLIP_BUFFER_SECONDS",    str(AGENT_CLIP_BUFFER_SECONDS)))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", str(AGENT_CONFIDENCE_THRESHOLD)))
IMAGE_QUALITY        = int(os.getenv("IMAGE_QUALITY",          str(AGENT_IMAGE_QUALITY)))
IMAGE_UPSCALE_FACTOR = float(os.getenv("IMAGE_UPSCALE_FACTOR", str(AGENT_IMAGE_UPSCALE_FACTOR)))
IMAGE_TARGET_RESOLUTION = os.getenv("IMAGE_TARGET_RESOLUTION", AGENT_IMAGE_TARGET_RESOLUTION)
IMAGE_FORMAT         = AGENT_IMAGE_FORMAT           # sourced from agent — not env-overridable
PHASE2_IMAGE_FORMAT  = AGENT_PHASE2_IMAGE_FORMAT    # sourced from agent — not env-overridable
IMAGE_INTERPOLATION  = os.getenv("IMAGE_INTERPOLATION", AGENT_IMAGE_INTERPOLATION).upper()
ENABLE_CROPPING      = os.getenv("ENABLE_CROPPING", str(AGENT_ENABLE_CROPPING)).lower() == "true"
ROTATION_ANGLE       = int(os.getenv("ROTATION_ANGLE", str(AGENT_ROTATION_ANGLE)))

# =============================================================================
# Web App Settings
# =============================================================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_webapp_dir / 'jobs.db'}")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(_webapp_dir / "uploads"))
RESULTS_DIR = os.getenv("RESULTS_DIR", str(_webapp_dir / "results"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))

# =============================================================================
# Langfuse Settings
# =============================================================================
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# =============================================================================
# Firebase Settings
# =============================================================================
# Set USE_FIREBASE=true to switch from local SQLite → Firebase Firestore.
# Leave it unset (or false) for local development with SQLite.
USE_FIREBASE = os.getenv("USE_FIREBASE", "false").lower() == "true"

# Path to the Firebase service account JSON file, OR the raw JSON string
# (useful for Hugging Face Spaces Secrets where you paste JSON as a string).
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")

# Firebase project ID (optional; detected automatically from service account)
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")

# Firebase Storage bucket (e.g. "my-project.appspot.com")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "")

# =============================================================================
# AWS S3 Settings (for persistent verification frame storage)
# =============================================================================
# Set USE_AWS_S3=true to upload Phase 2 verification frames to S3.
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME
# Free-tier safe: only PUT during analysis jobs; frames read via public URLs.
USE_AWS_S3 = os.getenv("USE_AWS_S3", "false").lower() == "true"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "")
AWS_S3_REGION = os.getenv("AWS_S3_REGION", "us-east-1")

# =============================================================================
# Paths
# =============================================================================
PIPELINE_SCRIPT_DIR = str(_parent_dir)

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
