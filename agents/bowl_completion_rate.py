"""
Bowl Completion Rate Analysis Pipeline with Two-Phase Detection
---------------------------------------------------------
A unified pipeline to detect bowl completion at the moment a customer
returns a ramen bowl to the chef/counter, and calculate the overall rate.

Phase 1: GPT-5-mini  - Detect bowl-return events and completion status
Phase 2: Gemini-2.5-pro - Verify completion status on short clips
"""

import sys
import os
import io
import re
import json
import base64
import logging
import shutil
import tempfile
import subprocess
import traceback
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Third-party imports
try:
    import cv2
    import numpy as np
    from PIL import Image
    from openai import OpenAI
    import google.genai as genai
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install requirements: pip install openai opencv-python-headless numpy pillow easyocr google-generativeai python-dotenv")
    sys.exit(1)

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None  # type: ignore[assignment]
    _EASYOCR_AVAILABLE = False
    print("Warning: easyocr not installed — timestamp OCR will be skipped. Run: pip install easyocr")

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# ⚙️ AGENT-LEVEL PARAMETER DEFAULTS
# Canonical defaults for the bowl completion rate task.
# webapp/config.py can import these when this agent is active.
# ============================================================================

AGENT_PHASE1_MODEL_NAME       = "gpt-5-mini"
AGENT_PHASE2_MODEL_NAME       = "gemini-2.5-pro"
AGENT_FPS                     = 1.0
AGENT_CONFIDENCE_THRESHOLD    = 0.8
AGENT_MAX_BATCH_SIZE_MB       = 35.0
AGENT_CLIP_BUFFER_SECONDS     = 3
AGENT_MAX_FRAMES_PER_BATCH    = 300
AGENT_BATCH_OVERLAP_FRAMES    = 5
AGENT_IMAGE_QUALITY           = 100
AGENT_IMAGE_UPSCALE_FACTOR    = 1.0
AGENT_IMAGE_TARGET_RESOLUTION = "auto"
AGENT_IMAGE_FORMAT            = "PNG"
AGENT_PHASE2_IMAGE_FORMAT     = "PNG"
AGENT_IMAGE_INTERPOLATION     = "LANCZOS"
AGENT_ENABLE_CROPPING         = True   # full-scene scan — no tight ROI crop
AGENT_ROTATION_ANGLE          = 0

# ============================================================================
# ⚙️ RUNTIME CONFIGURATION (env vars can override agent defaults above)
# ============================================================================

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PHASE1_MODEL_NAME = os.getenv("PHASE1_MODEL_NAME", AGENT_PHASE1_MODEL_NAME)
PHASE2_MODEL_NAME = os.getenv("PHASE2_MODEL_NAME", AGENT_PHASE2_MODEL_NAME)

# Analysis Mode Settings
ENABLE_PHASE2 = os.getenv("ENABLE_PHASE2", "true").lower() in ["true", "1", "yes"]

# Analysis Settings
FPS                  = float(os.getenv("FPS",                  str(AGENT_FPS)))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", str(AGENT_CONFIDENCE_THRESHOLD)))
MAX_BATCH_SIZE_MB    = float(os.getenv("MAX_BATCH_SIZE_MB",    str(AGENT_MAX_BATCH_SIZE_MB)))
CLIP_BUFFER_SECONDS  = int(os.getenv("CLIP_BUFFER_SECONDS",    str(AGENT_CLIP_BUFFER_SECONDS)))
MAX_FRAMES_PER_BATCH = int(os.getenv("MAX_FRAMES_PER_BATCH",   str(AGENT_MAX_FRAMES_PER_BATCH)))
BATCH_OVERLAP_FRAMES = int(os.getenv("BATCH_OVERLAP_FRAMES",   str(AGENT_BATCH_OVERLAP_FRAMES)))

# Image Settings
IMAGE_QUALITY           = int(os.getenv("IMAGE_QUALITY",          str(AGENT_IMAGE_QUALITY)))
IMAGE_UPSCALE_FACTOR    = float(os.getenv("IMAGE_UPSCALE_FACTOR",  str(AGENT_IMAGE_UPSCALE_FACTOR)))
IMAGE_TARGET_RESOLUTION = os.getenv("IMAGE_TARGET_RESOLUTION",    AGENT_IMAGE_TARGET_RESOLUTION)
IMAGE_FORMAT            = AGENT_IMAGE_FORMAT        # not env-overridable
PHASE2_IMAGE_FORMAT     = AGENT_PHASE2_IMAGE_FORMAT # always PNG — not env-overridable
PHASE1_MAX_LONG_EDGE: int = int(os.getenv("PHASE1_MAX_LONG_EDGE", "448"))
IMAGE_INTERPOLATION     = os.getenv("IMAGE_INTERPOLATION", AGENT_IMAGE_INTERPOLATION).upper()

# Cropping / Rotation
ENABLE_CROPPING = os.getenv("ENABLE_CROPPING", str(AGENT_ENABLE_CROPPING)).lower() == "true"
ROTATION_ANGLE  = int(os.getenv("ROTATION_ANGLE", str(AGENT_ROTATION_ANGLE)))

# Batch Processing Settings
BATCH_SIZE_SAFETY_MARGIN      = float(os.getenv("BATCH_SIZE_SAFETY_MARGIN",      "0.90"))
JSON_OVERHEAD_PER_FRAME_BYTES = int(os.getenv("JSON_OVERHEAD_PER_FRAME_BYTES",   "500"))

# Input/Output Defaults
DEFAULT_INPUT_VIDEO = os.getenv("INPUT_VIDEO_PATH", "")
DEFAULT_OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "./results_bowl_completion")

# ============================================================================
# 🤖 PROMPTS
# ============================================================================

PHASE1_SYSTEM_PROMPT="""# STRICT RAMEN BOWL COMPLETION ANALYSIS PROMPT

You are a **video analysis assistant specialized in in-store customer behavior analysis for ramen restaurants**.

Your task is to analyze video frames and determine whether a ramen bowl is **COMPLETED** or **NOT COMPLETED** **only at the exact moment** a customer **places the bowl back on the HIGHER silver-grey counter** while returning it.

## COUNTER LOCATION RULE (MANDATORY)
**ONLY detect events where the bowl is placed by the customer back on the HIGHER counter (silver-grey in color).**
- This counter is positioned above the lower eating counter, directly in front of the kitchen area.
- **DO NOT** detect or report events where the bowl is simply left on the lower eating counter or while the customer is still eating.
- The return event is defined by the physical act of lifting the bowl from the eating surface and setting it down on the elevated silver-grey ledge.

## CRITICAL VISIBILITY REQUIREMENT — NEAR-CAMERA BOWLS ONLY
- **ONLY analyze bowls that are clearly visible and positioned near the camera** (large, sharp, well-lit in the frame)
- **SKIP bowls that are distant, small, blurry, partially occluded, or where the interior contents cannot be clearly assessed**
- If the bowl is too far from the camera to estimate its remaining contents accurately, **do NOT report it**

## STRICT COMPLETION RULE (NO LEFTOVERS)
A ramen bowl is considered **COMPLETED** if and only if:
- **NO solid food remains in the bowl** (except for soup/broth and scattered green onions)
- The customer has finished all noodles, pork/chashu, and major toppings
- Only residual broth or a few stray green onions are left

A ramen bowl is **NOT COMPLETED** if:
- **ANY amount of solid food remains** (noodles, pork/chashu, bamboo shoots, or other toppings)
- Even if only a small portion of noodles or a single piece of pork is left, it is NOT COMPLETED

### ✅ COMPLETED (FINISHED):
- Only soup / broth remains (with or without floating green onions)
- Completely empty bowl with residual broth traces

### ❌ NOT COMPLETED (LEFTOVERS):
- **ANY noodles remain** in the bowl
- **ANY pork/chashu remains** in the bowl
- **ANY solid toppings remain** (bamboo shoots, egg, etc.)

## CRITICAL TEMPORAL RULE
- **ONLY evaluate the exact frame(s) where the customer physically places the bowl back on the HIGHER silver-grey counter**
- The timestamp must correspond **precisely** to the moment the bowl is set down on that specific ledge
- **IGNORE** all states where the bowl is on the lower eating counter, even if the customer seems finished

## OUTPUT FORMAT (JSON ONLY)
Return a JSON object with this exact structure:
{{"detections": [{{"video_timestamp": <seconds from video start as float>, "is_completed": "COMPLETED or NOT COMPLETED", "confidence": <0.0-1.0>, "remarks": "<briefly describe contents, e.g. 'returned to silver counter, empty of solids' or 'returned to silver counter, noodles remain' — include confirmation of counter location>"}}], "context_summary": "<brief summary of what was observed in this batch>"}}
"""

PHASE2_SYSTEM_PROMPT = """# PHASE 2: RAMEN BOWL COMPLETION VERIFICATION

You are a **verification assistant** for ramen bowl completion analysis.

Your task is to **verify** whether a previously detected bowl return event truly shows a **COMPLETED** or **NOT COMPLETED** ramen bowl, based on the **COUNTER LOCATION** and **STRICT COMPLETION CRITERIA**.

## COUNTER LOCATION RULE (MANDATORY)
**Only consider events where the bowl is placed by the customer back on the HIGHER counter (silver-grey in color).**
- This higher silver-grey counter is directly in front of the kitchen area.
- **DO NOT** report or verify events where the bowl is simply placed or left on the lower eating counter or while eating.
- If the bowl is NOT actively being placed on the higher silver-grey counter by the customer, mark `is_verified` as `false`.

## COMPLETION CRITERIA (STRICT: NO LEFTOVERS)
A ramen bowl is **COMPLETED** if and only if:
- **NO solid food remains** (except soup/green onions) at the moment it is placed on the higher silver-grey counter.
- All noodles, pork, and major toppings have been consumed.

A ramen bowl is **NOT COMPLETED** if:
- **ANY solid ingredients remain** (noodles, pork, bamboo shoots, etc.) when returned to the higher ledge.

### ✅ COMPLETED:
- Only soup / broth remains (with or without floating green onions)
- Completely empty bowl with residual broth

### ❌ NOT COMPLETED:
- **ANY amount of noodles, pork, or solid toppings** visibly present in the bowl when returned.

## REJECTION CRITERIA
Mark `is_verified: false` if:
- The bowl is **NOT** placed on the higher silver-grey counter (e.g., remains on the eating counter).
- The bowl is too far away or blurred to confirm contents.
- The return action is not clearly visible in the provided clip.

## OUTPUT FORMAT (JSON)
Respond with valid JSON only:
{
  "is_verified": true/false,
  "is_completed": "COMPLETED" or "NOT COMPLETED",
  "confidence": 0.0-1.0,
  "verification_notes": "detailed explanation — confirm returned to the HIGHER silver-grey counter, confirm if any solid food remains, and justify completion status"
}
"""

# ============================================================================
# 📝 LOGGING SETUP
# ============================================================================

# Fix console output encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _fh = logging.FileHandler('bowl_completion_analysis.log', encoding='utf-8', mode='w')
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_sh)
    logger.propagate = False


# ============================================================================
# 📦 DATA STRUCTURES
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration object."""
    input_video_path: str
    output_dir: str
    fps: float = FPS
    max_batch_size_mb: float = MAX_BATCH_SIZE_MB
    max_frames_per_batch: int = MAX_FRAMES_PER_BATCH
    clip_buffer_seconds: int = CLIP_BUFFER_SECONDS
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    batch_overlap_frames: int = BATCH_OVERLAP_FRAMES
    phase1_model_name: str = PHASE1_MODEL_NAME
    phase2_model_name: str = PHASE2_MODEL_NAME
    enable_phase2: bool = ENABLE_PHASE2
    enable_cropping: bool = ENABLE_CROPPING
    rotation_angle: int = ROTATION_ANGLE
    image_quality: int = IMAGE_QUALITY
    image_upscale_factor: float = IMAGE_UPSCALE_FACTOR
    image_target_resolution: str = IMAGE_TARGET_RESOLUTION
    image_format: str = IMAGE_FORMAT
    phase2_image_format: str = PHASE2_IMAGE_FORMAT
    image_interpolation: str = IMAGE_INTERPOLATION
    phase1_max_long_edge: int = PHASE1_MAX_LONG_EDGE
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    recording_hour: Optional[int] = None  # HH from recording timestamp; used for HH:MM:SS output

    def __post_init__(self):
        if self.openai_api_key is None:
            self.openai_api_key = OPENAI_API_KEY
        if self.google_api_key is None:
            self.google_api_key = GOOGLE_API_KEY

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env or environment variables")
        if not self.google_api_key and self.enable_phase2:
            logger.warning("GOOGLE_API_KEY is missing in .env! Phase 2 with Gemini will fail.")

        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class Detection:
    """Represents a ramen bowl completion detection."""
    timestamp: Optional[str]       # Real-world timestamp from frame OCR
    is_completed: str              # "COMPLETED" or "NOT COMPLETED"
    confidence: float              # 0.0 – 1.0
    remarks: str                   # Description of bowl contents
    video_timestamp: float = 0.0  # Video position in seconds
    video_source: str = ""        # Source video filename
    phase: int = 1                 # 1 = Phase 1, 2 = Phase 2 verified
    is_verified: bool = False
    verification_notes: str = ""


class FrameBatch:
    """Manages a batch of frames with size constraints."""

    def __init__(self, batch_id: int, max_size_mb: float, max_frames: int = MAX_FRAMES_PER_BATCH):
        self.batch_id = batch_id
        self.max_size_bytes = (max_size_mb * BATCH_SIZE_SAFETY_MARGIN) * 1024 * 1024
        self.max_frames = max_frames
        self.frames: List[Tuple[float, str]] = []
        self.current_size_bytes = 0

    def can_add_frame(self, frame_size: int) -> bool:
        """Check if frame can be added without exceeding size or count limit."""
        if len(self.frames) >= self.max_frames:
            return False
        frame_with_overhead = frame_size + JSON_OVERHEAD_PER_FRAME_BYTES
        return (self.current_size_bytes + frame_with_overhead) <= self.max_size_bytes

    def add_frame(self, timestamp: float, base64_frame: str, frame_size: int):
        """Add a frame to the batch."""
        if len(self.frames) >= self.max_frames:
            raise ValueError(f"Cannot add frame: batch already has {self.max_frames} frames")
        self.frames.append((timestamp, base64_frame))
        self.current_size_bytes += (frame_size + JSON_OVERHEAD_PER_FRAME_BYTES)

    def get_size_mb(self) -> float:
        return self.current_size_bytes / (1024 * 1024)

    def get_time_range(self) -> Tuple[float, float]:
        if not self.frames:
            return (0.0, 0.0)
        return (self.frames[0][0], self.frames[-1][0])


# ============================================================================
# 🧠 CORE PIPELINE
# ============================================================================

class BowlCompletionPipeline:
    """Main pipeline for bowl completion rate detection with two-phase verification."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Phase 1: OpenAI GPT
        self.openai_client = OpenAI(api_key=config.openai_api_key)

        # Phase 2: Gemini
        if config.google_api_key:
            try:
                self.gemini_client = genai.Client(api_key=config.google_api_key)
                logger.info("Gemini client initialized for Phase 2")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        else:
            self.gemini_client = None
            if config.enable_phase2:
                logger.warning("GOOGLE_API_KEY not set — Phase 2 will not work")

        self.context_history: List[str] = []
        self.phase1_detections: List[Detection] = []
        self.phase2_detections: List[Detection] = []
        self.temp_dir = tempfile.mkdtemp(prefix="bowl_completion_")
        self.video_fps: Optional[float] = None

        # ROI storage (kept for API compatibility with pipeline_adapter)
        self.roi = config.roi

        # Pre-compute ffmpeg path
        self._ffmpeg_path = shutil.which('ffmpeg')
        if not self._ffmpeg_path:
            for _candidate in [
                r'C:\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            ]:
                if os.path.exists(_candidate):
                    self._ffmpeg_path = _candidate
                    break

        # Pre-compute interpolation flag
        self._interpolation_flag = (
            cv2.INTER_LANCZOS4
            if config.image_interpolation.upper() == "LANCZOS"
            else cv2.INTER_CUBIC
        )
        _res = config.image_target_resolution
        if _res and _res.lower() != "auto":
            try:
                _parts = _res.split(",")
                self._target_size: Optional[Tuple[int, int]] = (int(_parts[0].strip()), int(_parts[1].strip()))
            except (ValueError, IndexError):
                logger.warning(f"Invalid target resolution format: {_res}. Using upscaled size.")
                self._target_size = None
        else:
            self._target_size = None

        # Token tracking
        self.token_usage: Dict[str, Any] = {
            "phase1_batches": [],
            "phase2_clips": [],
            "total_phase1_tokens": 0,
            "total_phase2_tokens": 0,
            "total_tokens": 0,
            "phase1_raw_responses": [],
            "phase2_raw_responses": [],
        }

        # Output subdirectories
        self.clips_dir     = os.path.join(config.output_dir, "phase2_clips")
        self.responses_dir = os.path.join(config.output_dir, "phase2_responses")
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)

        # OCR reader (lazy-loaded)
        self.ocr_reader = None

        logger.info(f"Initialized pipeline with temp dir: {self.temp_dir}")
        logger.info(f"Results will be saved to: {config.output_dir}")
        logger.info(f"Analysis Mode: {'Two-Phase' if config.enable_phase2 else 'Single-Phase'}")
        logger.info(f"Phase 1 Model: {config.phase1_model_name}")
        if config.enable_phase2:
            logger.info(f"Phase 2 Model: {config.phase2_model_name}")
        logger.info(f"Rotation Angle: {config.rotation_angle} degrees")
        logger.info(f"Upscale Factor: {config.image_upscale_factor}x")
        logger.info(f"Image Format: {config.image_format}")

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp dir: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir {self.temp_dir}: {e}")

    # =========================================================================
    # Frame Processing Methods
    # =========================================================================

    def rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame by the configured rotation angle."""
        if self.config.rotation_angle == 0:
            return frame
        elif self.config.rotation_angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotation_angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.config.rotation_angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            logger.warning(f"Invalid rotation angle: {self.config.rotation_angle}. Using 0 degrees.")
            return frame

    _MAX_UPSCALED_LONG_EDGE = 1920

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale frame by the configured factor, optionally clamped to target resolution."""
        if self.config.image_upscale_factor > 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.config.image_upscale_factor)
            new_h = int(h * self.config.image_upscale_factor)
            long_edge = max(new_w, new_h)
            if long_edge > self._MAX_UPSCALED_LONG_EDGE:
                scale = self._MAX_UPSCALED_LONG_EDGE / long_edge
                new_w = int(new_w * scale)
                new_h = int(new_h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=self._interpolation_flag)

        if self._target_size is not None:
            frame = cv2.resize(frame, self._target_size, interpolation=self._interpolation_flag)

        return frame

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to ROI region if ROI is set, otherwise return as-is."""
        if self.roi is None:
            return frame
        x1, y1, x2, y2 = self.roi
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return frame[y1:y2, x1:x2]

    def draw_roi_box(self, frame: np.ndarray) -> np.ndarray:
        """Draw a rectangle around the ROI on the full frame."""
        if self.roi is None:
            return frame
        frame_with_box = frame.copy()
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return frame_with_box

    def prepare_frame_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame: crop+upscale if cropping enabled, else draw ROI box."""
        if self.config.enable_cropping:
            cropped = self.crop_frame(frame)
            return self.upscale_frame(cropped)
        else:
            frame = self.upscale_frame(frame)
            return self.draw_roi_box(frame)

    def _compress_frame(self, frame: np.ndarray, format_override: Optional[str] = None, max_long_edge: Optional[int] = None) -> str:
        """Compress frame and convert to base64 using configured image format."""
        if max_long_edge is not None and max_long_edge > 0:
            h, w = frame.shape[:2]
            long_edge = max(h, w)
            if long_edge > max_long_edge:
                scale = max_long_edge / long_edge
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        image_format = format_override.upper() if format_override else self.config.image_format
        if image_format == "PNG":
            _, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
        return base64.b64encode(bytes(buf)).decode("utf-8")

    # =========================================================================
    # OCR — Timestamp Extraction
    # =========================================================================

    def _init_ocr_reader(self):
        """Initialize EasyOCR reader lazily."""
        if not _EASYOCR_AVAILABLE:
            return
        if self.ocr_reader is None:
            logger.info("Initializing OCR reader for timestamp extraction...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("OCR reader initialized")

    def extract_timestamp_from_frame(self, frame: np.ndarray) -> Optional[str]:
        """Extract timestamp from the bottom-left corner of the frame using OCR."""
        if not _EASYOCR_AVAILABLE:
            return None
        self._init_ocr_reader()
        height, width = frame.shape[:2]
        roi_height = int(height * 0.12)
        roi_width  = int(width * 0.35)
        roi = frame[height - roi_height:height, 0:roi_width]

        try:
            results = self.ocr_reader.readtext(roi, detail=0)
            if not results:
                return None
            full_text = ' '.join(results).strip()
            patterns = [
                r'\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}',
                r'\d{2}[-/]\d{2}[-/]\d{4}\s+\d{2}:\d{2}:\d{2}',
                r'\d{2}:\d{2}:\d{2}',
            ]
            for pattern in patterns:
                match = re.search(pattern, full_text)
                if match:
                    return match.group(0)
            return full_text if full_text else None
        except Exception as e:
            logger.warning(f"Error extracting timestamp from frame: {e}")
            return None

    def _extract_timestamp_at_video_time(self, video_path: str, video_timestamp: float) -> Optional[str]:
        """Extract OCR timestamp from the frame at the given video position."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video to extract timestamp at {video_timestamp}s")
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(video_timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                logger.warning(f"Could not read frame at {video_timestamp}s")
                return None
            return self.extract_timestamp_from_frame(frame)
        except Exception as e:
            logger.warning(f"Error extracting timestamp at video time {video_timestamp}s: {e}")
            return None

    # =========================================================================
    # Frame Extraction Methods
    # =========================================================================

    def extract_frames(self, video_path: str) -> List[Tuple[float, str]]:
        """Extract frames from video at configured FPS and convert to base64."""
        logger.info(f"Extracting frames from {video_path} at {self.config.fps} FPS")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps   = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration    = total_frames / video_fps
        self.video_fps = video_fps

        logger.info(f"Video: {video_fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")

        frames: List[Tuple[float, str]] = []
        time_interval    = 1.0 / self.config.fps
        next_extract_time = 0.0
        frame_count      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = frame_count / video_fps
            if current_time >= next_extract_time:
                rotated  = self.rotate_frame(frame)
                prepared = self.prepare_frame_for_analysis(rotated)
                b64      = self._compress_frame(prepared, max_long_edge=self.config.phase1_max_long_edge)
                frames.append((current_time, b64))
                if len(frames) % 100 == 0:
                    logger.info(f"Extracted {len(frames)} frames (timestamp: {current_time:.2f}s)")
                next_extract_time += time_interval
            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames total")
        return frames

    def extract_frames_phase2(self, video_path: str) -> List[Tuple[float, str]]:
        """Extract frames for Phase 2 verification — always PNG for lossless quality."""
        logger.info(f"Extracting Phase 2 frames from {video_path} at {self.config.fps} FPS (PNG format)")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: List[Tuple[float, str]] = []
        time_interval    = 1.0 / self.config.fps
        next_extract_time = 0.0
        frame_count      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = frame_count / video_fps
            if current_time >= next_extract_time:
                rotated  = self.rotate_frame(frame)
                prepared = self.prepare_frame_for_analysis(rotated)
                b64      = self._compress_frame(prepared, format_override=self.config.phase2_image_format)
                frames.append((current_time, b64))
                next_extract_time += time_interval
            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} Phase 2 frames (PNG)")
        return frames

    # =========================================================================
    # Batch Management
    # =========================================================================

    def create_batches(self, frames: List[Tuple[float, str]]) -> List[FrameBatch]:
        """Create batches of frames respecting size and count limits with overlap."""
        logger.info(f"Creating batches from {len(frames)} frames with {self.config.batch_overlap_frames} frames overlap")

        batches: List[FrameBatch] = []
        current_batch = FrameBatch(0, self.config.max_batch_size_mb, self.config.max_frames_per_batch)
        frame_index   = 0

        while frame_index < len(frames):
            timestamp, base64_frame = frames[frame_index]
            frame_size = len(base64_frame)

            if current_batch.can_add_frame(frame_size):
                current_batch.add_frame(timestamp, base64_frame, frame_size)
                frame_index += 1
            else:
                if current_batch.frames:
                    batches.append(current_batch)
                    logger.info(
                        f"Batch {current_batch.batch_id}: {len(current_batch.frames)} frames, "
                        f"{current_batch.get_size_mb():.2f} MB, "
                        f"time range: {current_batch.frames[0][0]:.1f}s - {current_batch.frames[-1][0]:.1f}s"
                    )
                    new_batch = FrameBatch(len(batches), self.config.max_batch_size_mb, self.config.max_frames_per_batch)
                    if self.config.batch_overlap_frames > 0:
                        overlap_start = max(0, len(current_batch.frames) - self.config.batch_overlap_frames)
                        for ts, b64 in current_batch.frames[overlap_start:]:
                            if new_batch.can_add_frame(len(b64)):
                                new_batch.add_frame(ts, b64, len(b64))
                            else:
                                logger.warning(f"Overlap frame at {ts:.2f}s too large, skipping")
                                break
                    current_batch = new_batch
                else:
                    logger.error(f"Frame at {timestamp:.2f}s too large ({frame_size/1024/1024:.2f} MB) — skipping")
                    frame_index += 1

        if current_batch.frames:
            batches.append(current_batch)
            logger.info(
                f"Batch {current_batch.batch_id}: {len(current_batch.frames)} frames, "
                f"{current_batch.get_size_mb():.2f} MB, "
                f"time range: {current_batch.frames[0][0]:.1f}s - {current_batch.frames[-1][0]:.1f}s"
            )

        logger.info(f"Created {len(batches)} batches total")
        return batches

    # =========================================================================
    # Phase 1 Analysis
    # =========================================================================

    def analyze_batch_phase1(self, batch: FrameBatch, context: str, retry_count: int = 0) -> Dict[str, Any]:
        """Analyze a batch with automatic retry on size errors."""
        MAX_RETRIES = 2
        logger.info(f"Analyzing batch {batch.batch_id} (Phase 1) — {len(batch.frames)} frames, {batch.get_size_mb():.2f} MB")

        schema_instruction = 'Respond with JSON only (no markdown): {"detections":[{"video_timestamp":number,"is_completed":"COMPLETED|NOT COMPLETED","confidence":number,"remarks":string}],"context_summary":string}'

        mime_type = "image/png" if self.config.image_format == "PNG" else "image/jpeg"

        content: List[Any] = [
            {"type": "text", "text": PHASE1_SYSTEM_PROMPT.format(context=context)},
            {"type": "text", "text": schema_instruction},
        ]

        for timestamp, base64_frame in batch.frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_frame}", "detail": "high"},
            })
            content.append({"type": "text", "text": f"t={timestamp:.1f}"})

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.phase1_model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=4096,
            )
            raw_content = response.choices[0].message.content
            logger.info("=" * 60)
            logger.info(f"[DEBUG] BATCH {batch.batch_id} - RAW API RESPONSE:")
            logger.info(f"{raw_content}")
            logger.info("=" * 60)
            print(f"\n{'=' * 60}")
            print(f"[DEBUG] BATCH {batch.batch_id} - RAW API RESPONSE:")
            print(f"{raw_content}")
            print(f"{'=' * 60}\n")
            self.token_usage["phase1_raw_responses"].append(
                f"=== Batch {batch.batch_id} ===\n{raw_content}"
            )

            result = json.loads(raw_content)

            raw_detections = result.get("detections", [])
            logger.info(f"[DEBUG] Batch {batch.batch_id}: Found {len(raw_detections)} raw detections before filtering")
            print(f"[DEBUG] Batch {batch.batch_id}: Found {len(raw_detections)} raw detections before filtering")

            for i, det in enumerate(raw_detections):
                confidence = det.get("confidence", 0) or 0
                video_ts = det.get("video_timestamp", "N/A")
                is_completed = det.get("is_completed", "N/A")
                remarks = det.get("remarks", "")
                filter_status = "PASSED" if confidence >= self.config.confidence_threshold else f"FILTERED (conf {confidence} < threshold {self.config.confidence_threshold})"
                logger.info(f"  Detection {i+1}: ts={video_ts}, completed={is_completed}, conf={confidence}, remarks={remarks}, status={filter_status}")
                print(f"  Detection {i+1}: ts={video_ts}, completed={is_completed}, conf={confidence}, remarks={remarks}, status={filter_status}")

            usage = response.usage
            self.token_usage["phase1_batches"].append({
                "batch_id": batch.batch_id,
                "frames": len(batch.frames),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            })
            self.token_usage["total_phase1_tokens"] += usage.total_tokens
            self.token_usage["total_tokens"] += usage.total_tokens

            logger.info(
                f"Batch {batch.batch_id} complete | tokens={usage.total_tokens} | "
                f"detections={len(result.get('detections', []))}"
            )
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error analyzing batch {batch.batch_id}: {error_msg}")

            if ("400" in error_msg or "request" in error_msg.lower()) and retry_count < MAX_RETRIES:
                logger.warning(f"Batch {batch.batch_id} too large, splitting and retrying...")
                mid_point = len(batch.frames) // 2
                if mid_point < 1:
                    logger.error(f"Cannot split batch {batch.batch_id} further — single frame too large")
                    return {"detections": [], "context_summary": "Error: Payload too large"}

                batch1 = FrameBatch(batch.batch_id,       self.config.max_batch_size_mb, max_frames=mid_point)
                batch2 = FrameBatch(batch.batch_id + 0.5, self.config.max_batch_size_mb, max_frames=mid_point)
                for i, (ts, b64) in enumerate(batch.frames):
                    if i < mid_point:
                        batch1.add_frame(ts, b64, len(b64))
                    else:
                        batch2.add_frame(ts, b64, len(b64))

                result1 = self.analyze_batch_phase1(batch1, context, retry_count + 1)
                result2 = self.analyze_batch_phase1(batch2, result1.get("context_summary", context), retry_count + 1)
                return {
                    "detections": result1.get("detections", []) + result2.get("detections", []),
                    "context_summary": result2.get("context_summary", ""),
                }

            traceback.print_exc()
            return {"detections": [], "context_summary": f"Error: {error_msg}"}

    # =========================================================================
    # Video Clipping
    # =========================================================================

    def clip_video_segment(self, video_path: str, start_time: float, end_time: float, output_path: str):
        """Clip a segment from video using ffmpeg, adding buffer padding."""
        start_with_buffer = max(0, start_time - self.config.clip_buffer_seconds)
        end_with_buffer   = end_time + self.config.clip_buffer_seconds
        duration          = end_with_buffer - start_with_buffer

        ffmpeg_path = self._ffmpeg_path
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found. Please install FFmpeg to use this pipeline.")

        cmd = [
            ffmpeg_path, '-i', video_path,
            '-ss', str(start_with_buffer), '-t', str(duration),
            '-c', 'copy', '-y', output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Saved clip: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error clipping video: {e.stderr}")
            raise

    # =========================================================================
    # Phase 2 Verification
    # =========================================================================

    def analyze_clip_phase2(
        self,
        clip_frames: List[Tuple[float, str]],
        detection: Detection,
        clip_index: int,
    ) -> Dict[str, Any]:
        """Verify a detection using Gemini on a short clip around the event."""
        logger.info(f"Verifying detection {clip_index} with Gemini model: {self.config.phase2_model_name}")

        if not self.gemini_client:
            logger.error("Gemini client not initialized. Check GOOGLE_API_KEY.")
            return {
                "is_verified": False,
                "is_completed": "NOT COMPLETED",
                "confidence": 0.0,
                "verification_notes": "Gemini client not initialized",
            }

        content: List[Any] = [
            (
                f"Verify bowl completion detection. "
                f"Phase 1 detected: {detection.is_completed} (confidence: {detection.confidence:.2f}). "
                f"Remarks: {detection.remarks}. "
                f"Analyzing {len(clip_frames)} frames around the detection moment."
            )
        ]

        for timestamp, base64_frame in clip_frames:
            try:
                image_data = base64.b64decode(base64_frame)
                image      = Image.open(io.BytesIO(image_data))
                content.append(f"t={timestamp:.1f}")
                content.append(image)
            except Exception as e:
                logger.warning(f"Failed to process frame at {timestamp}s: {e}")

        try:
            response = self.gemini_client.models.generate_content(
                model=self.config.phase2_model_name,
                contents=content,
                config={
                    "response_mime_type": "application/json",
                    "system_instruction": PHASE2_SYSTEM_PROMPT,
                },
            )
            result_text = response.text
            self.token_usage["phase2_raw_responses"].append(
                f"=== Clip {clip_index} ===\n{result_text}"
            )
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(result_text)

            # Track token usage
            try:
                if hasattr(response, 'usage_metadata'):
                    tokens_used = response.usage_metadata.total_token_count
                    self.token_usage["phase2_clips"].append({
                        "clip_index": clip_index,
                        "frames": len(clip_frames),
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": tokens_used,
                    })
                    self.token_usage["total_phase2_tokens"] += tokens_used
                    self.token_usage["total_tokens"]        += tokens_used
                    logger.info(f"Clip {clip_index} verified | tokens={tokens_used} | result={result.get('is_verified')}")
            except Exception as token_err:
                logger.warning(f"Could not extract token usage from Gemini response: {token_err}")

            return result

        except Exception as e:
            logger.error(f"Error in Phase 2 analysis (Gemini): {e}")
            traceback.print_exc()
            return {
                "is_verified": False,
                "is_completed": "NOT COMPLETED",
                "confidence": 0.0,
                "verification_notes": f"Gemini Error: {e}",
            }

    # =========================================================================
    # Phase Runners
    # =========================================================================

    def run_phase1(self, video_path: str) -> List[Detection]:
        """Run Phase 1: scan entire video and detect bowl return events."""
        logger.info(f"PHASE 1: Scanning {video_path}")
        all_frames = self.extract_frames(video_path)
        batches    = self.create_batches(all_frames)

        detections: List[Detection] = []
        context = "This is the first batch of the video."

        for batch in batches:
            result = self.analyze_batch_phase1(batch, context)
            for det in result.get("detections", []):
                if float(det.get("confidence", 0)) >= self.config.confidence_threshold:
                    video_ts = float(det.get("video_timestamp", 0.0))
                    # Prefer formula-based timestamp (recording_hour + video seconds → HH:MM:SS)
                    if self.config.recording_hour is not None:
                        total_s = video_ts + self.config.recording_hour * 3600
                        real_ts = f"{int(total_s // 3600):02d}:{int((total_s % 3600) // 60):02d}:{int(total_s % 60):02d}"
                    else:
                        real_ts = self._extract_timestamp_at_video_time(video_path, video_ts)
                    detections.append(Detection(
                        timestamp=real_ts,
                        is_completed=det.get("is_completed", "NOT COMPLETED"),
                        confidence=float(det["confidence"]),
                        remarks=det.get("remarks", ""),
                        video_timestamp=video_ts,
                        video_source=Path(video_path).name,
                    ))
            context_summary = result.get("context_summary", "")
            if context_summary:
                context = context_summary

        logger.info(f"Phase 1 complete: {len(detections)} detections found")
        self.phase1_detections = detections
        return detections

    def run_phase2(self, video_path: str, phase1_detections: List[Detection]) -> List[Detection]:
        """Run Phase 2: verify each Phase 1 detection with a short Gemini clip check."""
        logger.info(f"PHASE 2: Verifying {len(phase1_detections)} detections")
        verified: List[Detection] = []

        for idx, detection in enumerate(phase1_detections, 1):
            logger.info(
                f"Verifying {idx}/{len(phase1_detections)}: "
                f"{detection.is_completed} at {detection.video_timestamp:.1f}s "
                f"(confidence: {detection.confidence:.2f})"
            )

            clip_filename = f"clip_{idx}_{detection.video_timestamp:.1f}s.mp4"
            clip_path     = os.path.join(self.clips_dir, clip_filename)

            self.clip_video_segment(
                video_path,
                detection.video_timestamp,
                detection.video_timestamp,
                clip_path,
            )

            clip_frames = self.extract_frames_phase2(clip_path)

            # ── Save verification frames to disk for UI display ──────────────
            det_frames_dir = os.path.join(
                self.config.output_dir, "phase2_frames", f"detection_{idx:03d}"
            )
            os.makedirs(det_frames_dir, exist_ok=True)
            for fi, (ts, b64_frame) in enumerate(clip_frames):
                try:
                    frame_data = base64.b64decode(b64_frame)
                    frame_path = os.path.join(det_frames_dir, f"frame_{fi:04d}_{ts:.1f}s.png")
                    with open(frame_path, "wb") as _ff:
                        _ff.write(frame_data)
                except Exception as _fe:
                    logger.warning(f"Could not save frame {fi} for detection {idx}: {_fe}")

            result      = self.analyze_clip_phase2(clip_frames, detection, idx)

            # Save raw verification response
            response_file = os.path.join(self.responses_dir, f"verification_{idx}.json")
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "phase1_detection": {
                        "timestamp": detection.timestamp,
                        "is_completed": detection.is_completed,
                        "confidence": detection.confidence,
                        "remarks": detection.remarks,
                        "video_timestamp": detection.video_timestamp,
                    },
                    "phase2_verification": result,
                }, f, indent=2)

            if result.get("is_verified", False):
                verified_det = Detection(
                    timestamp=detection.timestamp,
                    is_completed=result.get("is_completed", detection.is_completed),
                    confidence=float(result.get("confidence", detection.confidence)),
                    remarks=result.get("verification_notes", detection.remarks),
                    video_timestamp=detection.video_timestamp,
                    video_source=detection.video_source,
                    phase=2,
                    is_verified=True,
                    verification_notes=result.get("verification_notes", ""),
                )
                verified.append(verified_det)
                logger.info(f"[✓] VERIFIED: {verified_det.is_completed} (confidence: {verified_det.confidence:.2f})")
            else:
                logger.info(f"[✗] REJECTED: {result.get('verification_notes', 'Failed verification')}")

        logger.info(f"Phase 2 complete: {len(verified)}/{len(phase1_detections)} detections verified")
        self.phase2_detections = verified
        return verified

    # =========================================================================
    # Output
    # =========================================================================

    def create_merged_video(self, video_path: str):
        """Create a merged video of all verified clips."""
        if not self.phase2_detections:
            logger.info("No verified detections — skipping merged video creation")
            return

        logger.info(f"Creating merged video from {len(self.phase2_detections)} verified clips")

        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for idx, detection in enumerate(self.phase2_detections, 1):
                clip_temp = os.path.join(self.temp_dir, f"merge_clip_{idx}.mp4")
                self.clip_video_segment(video_path, detection.video_timestamp, detection.video_timestamp, clip_temp)
                f.write(f"file '{clip_temp.replace(os.sep, '/')}'\n")

        merged_path = os.path.join(self.config.output_dir, "merged_verified_clips.mp4")
        ffmpeg_path = self._ffmpeg_path

        if ffmpeg_path and os.path.exists(ffmpeg_path):
            try:
                subprocess.run([
                    ffmpeg_path, '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-c', 'copy', '-y', merged_path,
                ], check=True, capture_output=True)
                logger.info(f"Merged video created: {merged_path}")
            except Exception as e:
                logger.error(f"Failed to create merged video: {e}")

    def save_results(self):
        """Save analysis results to JSON + text summary."""
        # Choose the final detection set
        if self.config.enable_phase2 and self.phase2_detections:
            final_detections = self.phase2_detections
            analysis_type    = "Phase 2 Verified"
        else:
            final_detections = self.phase1_detections
            analysis_type    = "Phase 1"

        # Completion statistics
        total_cases      = len(final_detections)
        completed_cases  = sum(1 for d in final_detections if d.is_completed == "COMPLETED")
        completion_rate  = round((completed_cases / total_cases) * 100, 2) if total_cases > 0 else 0.0

        # Build detections output list
        detections_output = []
        for d in final_detections:
            det_dict: Dict[str, Any] = {
                "timestamp":       d.timestamp or "N/A",
                "is_completed":    d.is_completed,
                "confidence":      round(d.confidence, 2),
                "remarks":         d.remarks,
                "video_source":    d.video_source,
                "video_timestamp": round(d.video_timestamp, 2),
            }
            if d.phase == 2:
                det_dict["phase"]              = 2
                det_dict["is_verified"]        = d.is_verified
                det_dict["verification_notes"] = d.verification_notes
            detections_output.append(det_dict)

        # JSON results
        results = {
            "metadata": {
                "video":                 self.config.input_video_path,
                "analysis_date":         datetime.now().isoformat(),
                "analysis_type":         analysis_type,
                "phase1_detections":     len(self.phase1_detections),
                "phase2_verified":       len(self.phase2_detections) if self.config.enable_phase2 else "N/A",
                "total_final_detections": total_cases,
            },
            "completion_statistics": {
                "total_cases":       total_cases,
                "completed_cases":   completed_cases,
                "not_completed_cases": total_cases - completed_cases,
                "completion_rate":   completion_rate,
            },
            "detections": detections_output,
        }

        output_path = os.path.join(self.config.output_dir, "ramen_completion_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")

        # Token usage JSON
        token_path = os.path.join(self.config.output_dir, "token_usage.json")
        with open(token_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_usage, f, indent=2)
        logger.info(f"Token usage saved to: {token_path}")
        logger.info(f"Total tokens used: {self.token_usage['total_tokens']}")

        # Text summary
        summary_path = os.path.join(self.config.output_dir, "analysis_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"RAMEN BOWL COMPLETION ANALYSIS SUMMARY\n{'='*70}\n")
            f.write(f"Video: {self.config.input_video_path}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Type: {analysis_type}\n")
            if self.config.enable_phase2:
                f.write(f"Phase 1 Detections: {len(self.phase1_detections)}\n")
                f.write(f"Phase 2 Verified: {len(self.phase2_detections)}\n")
            f.write(f"Final Detections: {total_cases}\n")
            f.write(f"Total Tokens Used: {self.token_usage['total_tokens']}\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"COMPLETION STATISTICS\n{'-'*70}\n")
            f.write(f"Total Cases: {total_cases}\n")
            f.write(f"Completed Cases: {completed_cases}\n")
            f.write(f"Not Completed Cases: {total_cases - completed_cases}\n")
            f.write(f"Overall Completion Rate: {completion_rate}%\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"CONFIGURATION PARAMETERS\n{'-'*70}\n")
            f.write(f"Phase 1 Model: {self.config.phase1_model_name}\n")
            f.write(f"Phase 2 Model: {self.config.phase2_model_name}\n")
            f.write(f"Enable Phase 2: {self.config.enable_phase2}\n")
            f.write(f"FPS: {self.config.fps}\n")
            f.write(f"Confidence Threshold: {self.config.confidence_threshold}\n")
            f.write(f"Max Batch Size (MB): {self.config.max_batch_size_mb}\n")
            f.write(f"Max Frames Per Batch: {self.config.max_frames_per_batch}\n")
            f.write(f"Batch Overlap Frames: {self.config.batch_overlap_frames}\n")
            f.write(f"Clip Buffer (seconds): {self.config.clip_buffer_seconds}\n")
            f.write(f"Image Quality: {self.config.image_quality}\n")
            f.write(f"Image Format: {self.config.image_format}\n")
            f.write(f"Upscale Factor: {self.config.image_upscale_factor}x\n")
            f.write(f"Rotation Angle: {self.config.rotation_angle} degrees\n")
            f.write(f"{'='*70}\n\n")

            header = "VERIFIED DETECTIONS (Phase 2)" if self.config.enable_phase2 else "DETECTIONS (Phase 1)"
            f.write(f"{header}\n{'-'*70}\n\n")

            if not final_detections:
                f.write("No detections found.\n")
            else:
                for idx, d in enumerate(final_detections, 1):
                    f.write(f"{idx}. Timestamp: {d.timestamp or 'N/A'}\n")
                    f.write(f"   Video Time: {d.video_timestamp:.2f}s\n")
                    f.write(f"   Status: {d.is_completed}\n")
                    f.write(f"   Confidence: {d.confidence:.2f}\n")
                    f.write(f"   Remarks: {d.remarks}\n")
                    if d.phase == 2 and d.verification_notes:
                        f.write(f"   Verification: {d.verification_notes}\n")
                    f.write(f"   Video: {d.video_source}\n\n")

        logger.info(f"Summary saved to: {summary_path}")

    # =========================================================================
    # Main Run
    # =========================================================================

    def run(self):
        """Main pipeline execution."""
        try:
            self.run_phase1(self.config.input_video_path)

            if self.config.enable_phase2:
                logger.info("Two-phase analysis mode: Running Phase 2 verification...")
                if self.phase1_detections:
                    self.run_phase2(self.config.input_video_path, self.phase1_detections)
                    self.create_merged_video(self.config.input_video_path)
                else:
                    logger.info("No Phase 1 detections found. Skipping Phase 2.")
                    self.phase2_detections = []
            else:
                logger.info("Single-phase analysis mode: Skipping Phase 2 verification.")
                self.phase2_detections = self.phase1_detections

            self.save_results()

            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"Phase 1 Detections: {len(self.phase1_detections)}")
            print(f"Phase 2 Verified: {len(self.phase2_detections)}")
            print(f"Results saved to: {self.config.output_dir}")
            print("="*60)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()


# ============================================================================
# 🚀 ENTRY POINT
# ============================================================================

def get_unique_output_dir(base_dir: str, video_name: str) -> str:
    """Generate a unique output directory path."""
    base_path = Path(base_dir) / video_name
    if not base_path.exists():
        return str(base_path)
    counter = 1
    while True:
        new_path = Path(base_dir) / f"{video_name}_{counter}"
        if not new_path.exists():
            return str(new_path)
        counter += 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ramen Bowl Completion Rate Analysis Pipeline")
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--fps",       type=float, default=FPS,    help="Frames per second to analyze")
    parser.add_argument("--no-phase2", action="store_true",        help="Disable Phase 2 verification")
    args = parser.parse_args()

    video_path = args.video_path or DEFAULT_INPUT_VIDEO

    if not video_path:
        print("Error: No video path provided.")
        print("Usage: python bowl_completion_rate.py <video_path>")
        print("OR set INPUT_VIDEO_PATH in .env file")
        return

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set in .env file!")
        return

    if not GOOGLE_API_KEY and not args.no_phase2 and ENABLE_PHASE2:
        print("\n[WARNING] GOOGLE_API_KEY is not set in .env!")
        print("Phase 2 analysis will fail. Add --no-phase2 to run single-phase analysis.")
        print("-" * 60)

    video_name = Path(video_path).stem
    output_dir = get_unique_output_dir(args.output, video_name)

    config = PipelineConfig(
        input_video_path=video_path,
        output_dir=output_dir,
        fps=args.fps,
        enable_phase2=not args.no_phase2 and ENABLE_PHASE2,
    )

    pipeline = BowlCompletionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
