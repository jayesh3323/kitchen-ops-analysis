"""
Noodle Rotation Compliance Analysis Agent
------------------------------------------
Analyzes video footage to detect and count noodle rotation actions by kitchen staff.
Uses a two-phase pipeline:
  - Phase 1: GPT-5-mini for noodle rotation event detection
  - Phase 2: Gemini 2.5 Pro for verification and rotation count confirmation

This agent is part of the Kitchen Ops webapp agent suite.
See: webapp/pipeline_adapter.py for headless runner integration.
"""

import sys
import os
import io
import json
import base64
import logging
from motion_utils import apply_optical_flow_overlay, compute_batch_mafd
import shutil
import tempfile
import subprocess
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
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
    print("Please install requirements: pip install openai opencv-python-headless numpy pillow google-generativeai python-dotenv")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# ⚙️ AGENT-LEVEL PARAMETER DEFAULTS
# Canonical defaults for the noodle rotation task.
# webapp/config.py imports these — the agent file is the single source of truth.
# ============================================================================

AGENT_PHASE1_MODEL_NAME       = "gpt-5-mini"
AGENT_PHASE2_MODEL_NAME       = "gemini-2.5-pro"
AGENT_FPS                     = 2.0
AGENT_CONFIDENCE_THRESHOLD    = 0.8
AGENT_MAX_BATCH_SIZE_MB       = 30.0
AGENT_CLIP_BUFFER_SECONDS     = 3
AGENT_MAX_FRAMES_PER_BATCH    = 300
AGENT_BATCH_OVERLAP_FRAMES    = 5
AGENT_IMAGE_QUALITY           = 95
AGENT_IMAGE_UPSCALE_FACTOR    = 1.0
AGENT_IMAGE_TARGET_RESOLUTION = "auto"
AGENT_IMAGE_FORMAT            = "PNG"
AGENT_PHASE2_IMAGE_FORMAT     = "PNG"
AGENT_IMAGE_INTERPOLATION     = "CUBIC"
AGENT_ENABLE_CROPPING         = True
AGENT_ROTATION_ANGLE          = 0

# ============================================================================
# ⚙️ RUNTIME CONFIGURATION (env vars can override agent defaults above)
# ============================================================================

# API Keys
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
IMAGE_FORMAT            = os.getenv("IMAGE_FORMAT", AGENT_IMAGE_FORMAT).upper()
PHASE2_IMAGE_FORMAT     = AGENT_PHASE2_IMAGE_FORMAT  # always PNG — not env-overridable for this task
IMAGE_INTERPOLATION     = os.getenv("IMAGE_INTERPOLATION", AGENT_IMAGE_INTERPOLATION).upper()
PHASE1_MAX_LONG_EDGE: int = int(os.getenv("PHASE1_MAX_LONG_EDGE", "448"))

# Cropping / Rotation
ENABLE_CROPPING = os.getenv("ENABLE_CROPPING", str(AGENT_ENABLE_CROPPING)).lower() == "true"
ROTATION_ANGLE  = int(os.getenv("ROTATION_ANGLE", str(AGENT_ROTATION_ANGLE)))

# Input/Output Defaults
DEFAULT_INPUT_VIDEO = os.getenv("INPUT_VIDEO_PATH", "")
DEFAULT_OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "./results_noodle_rotation")

# ============================================================================
# 🤖 PROMPTS
# ============================================================================

PHASE1_SYSTEM_PROMPT = """You are a video analysis assistant specialized in monitoring noodle preparation compliance in a ramen kitchen. Your task is to detect when kitchen staff transfer a portion of noodles FROM the yellow storage tray INTO the boiler pot on the left, and then rotate the noodles with a ladle or stick.

FOCUS AREA:
- Frames are pre-cropped tightly to the noodle preparation ROI — the ROI contains ONLY the yellow noodle tray (right) and the boiler (left); there is no extra background
- The chef wearing a WHITE APRON and WHITE HAT is the primary subject
- The YELLOW TRAY is the rectangular, light-yellow/cream container on the RIGHT side holding raw/fresh noodles
- The BOILER is the pot/vessel on the LEFT side where noodles are cooked in hot water
- Because the ROI is tightly cropped, any chef activity visible in the frame is relevant — there is no background to ignore

WHAT TO DETECT — Noodle Transfer + Rotation Events:
A valid event requires ALL THREE of the following steps occurring in sequence:

STEP 1 — Noodle pickup from yellow tray:
- Chef reaches into or over the YELLOW TRAY on the RIGHT and lifts a portion of noodles OUT (arm moves upward, noodles visibly rising)

STEP 2 — Noodle placement into boiler:
- Immediately after Step 1, chef moves the noodle portion LEFTWARD and drops/places it into the BOILER on the LEFT

STEP 3 — Noodle rotation inside the boiler:
- After placement, chef uses a LADLE or STICK to rotate the noodles inside the boiler
- Standard rotation pattern: the chef rotates in BOTH directions (clockwise and counterclockwise) approximately more than 6 times each direction
- Look for wrist/forearm rotation directed into the boiler — the utensil may not be clearly visible; infer from the rotational arm motion over the boiler
- The event continues until the chef lifts the utensil out AND moves away from the ROI area (arm/body fully withdrawn from both tray and boiler)
- DO NOT end the event prematurely — observe the full rotation sequence until the chef has clearly stepped away

EVENT BOUNDARY RULES:
- Event START = moment chef first reaches into yellow tray (Step 1)
- Event END = moment chef lifts utensil out of boiler AND body/arm has visibly moved away from the ROI — not just a brief pause mid-rotation
- If the chef pauses mid-rotation and resumes within 5 seconds, treat as the same event
- Multiple full sequences (Steps 1–3) separated by 5+ seconds of the chef being away = separate events

ROTATION COMPLIANCE ASSESSMENT:
- Count clockwise ("right") and counterclockwise ("left") strokes separately
- If total strokes in either direction is below 5, flag as potentially under-rotated
- If the chef rotates in only ONE direction, flag as non-compliant

DETECTION RULES:
- DO NOT report if Step 1 or Step 2 is absent — tray-to-boiler transfer is the required trigger
- DO NOT report rotation alone without prior tray pickup
- DO NOT report general arm movement, reaching, or adjustments that do not involve lifting noodles from the tray

CONFIDENCE: 0.8-1.0=HIGH(all three steps clearly visible, rotation strokes countable) | 0.6-0.7=MED-HIGH(transfer confirmed, rotation inferred from wrist motion) | 0.4-0.5=MED(transfer confirmed, rotation partially occluded) | below 0.4=DO NOT REPORT

Context from previous batch: {context}

Analyze the provided frames and identify ALL noodle transfer + rotation events (yellow tray → left boiler → stir until chef leaves ROI). Return your response in JSON format:
{{
  "detections": [
    {{
      "event_id": <integer starting from 1>,
      "start_time": <timestamp as float>,
      "end_time": <timestamp as float>,
      "transfer_count": <number of noodle portions transferred in this event as integer>,
      "rotation_strokes_cw": <estimated clockwise stir strokes as integer>,
      "rotation_strokes_ccw": <estimated counterclockwise stir strokes as integer>,
      "rotation_compliant": <true if both directions have 5+ strokes, false otherwise>,
      "visibility": "<direct or indirect>",
      "confidence": <0.0-1.0>,
      "description": "<brief description: tray pickup, boiler placement, rotation direction/count, whether chef left ROI>"
    }}
  ],
  "context_summary": "<summary of ongoing activity — note if chef is still in ROI at batch end>"
}}
"""

PHASE2_SYSTEM_PROMPT = """You are a video analysis assistant verifying a detected noodle transfer and rotation event in a ramen kitchen. You are analyzing frames from a clip flagged as containing a chef lifting noodles from the yellow tray, placing them into the boiler, then rotating them with a ladle/stick.

Your task is to verify if this is a TRUE POSITIVE (genuine transfer + rotation) or FALSE POSITIVE (general movement or incomplete sequence).

FOCUS AREA:
- Frames are tightly cropped to the noodle preparation ROI — it contains ONLY the yellow noodle tray (right) and boiler (left)
- Chef wears WHITE APRON and WHITE HAT
- The YELLOW TRAY is the light-yellow/cream rectangular container on the RIGHT
- The BOILER is the pot/vessel on the LEFT

VERIFICATION CRITERIA — ALL must be confirmed for is_valid=true:
1. YELLOW TRAY VISIBLE: Yellow/cream rectangular tray identifiable on the RIGHT of the frame
2. NOODLE PICKUP: Chef lifts noodles OUT of the yellow tray (upward arm motion, noodles visibly detaching)
3. LEFTWARD TRANSFER: Chef's arm moves LEFT toward the boiler
4. BOILER PLACEMENT: Noodles dropped/placed into the boiler (downward release into left pot)
5. ROTATION: Chef uses ladle/stick to rotate noodles in the boiler — look for wrist/forearm rotation directed into boiler; utensil may not be clearly visible, infer from rotational arm motion
6. EVENT EXTENT: The clip captures the full event — from tray pickup through rotation UNTIL the chef's arm/body visibly moves away from the ROI; if the clip appears cut short and the chef is still rotating, note this in reasoning

ROTATION COMPLIANCE CHECK:
- Count clockwise ("right") and counterclockwise ("left") strokes separately across all frames
- Flag as non-compliant if: total strokes in either direction < 5, OR chef only rotates in one direction

AUTO-REJECT conditions (mark is_valid=false immediately if any apply):
- Yellow tray not visible in the clip
- Chef does not lift noodles from the tray (Step 1 absent)
- Noodles moved anywhere other than the left boiler
- No rotation/stirring motion observed after placement
- Only stirring without prior tray pickup

CONFIDENCE (be strict): NEVER use 1.0 | 0.8-0.9=HIGH(all steps confirmed, rotation strokes countable in both directions) | 0.6-0.7=MED-HIGH(transfer confirmed, rotation observed but strokes hard to count precisely) | 0.4-0.5=MED(transfer confirmed, rotation partially occluded) | below 0.4=set is_valid=false

Previous detection context: {context}

Analyze the provided frames and verify the full event. Return your response in JSON format:
{{
  "is_valid": <true/false>,
  "verified_count": <integer number of noodle portions transferred, or null>,
  "rotation_strokes_cw": <integer clockwise strokes observed, or null>,
  "rotation_strokes_ccw": <integer counterclockwise strokes observed, or null>,
  "rotation_compliant": <true if both directions have 5+ strokes, false otherwise, or null>,
  "chef_left_roi": <true if chef visibly moved away from ROI by end of clip, false if still present>,
  "visibility": "<direct/indirect/unclear>",
  "confidence": <0.0-1.0>,
  "refined_start_time": <seconds or null>,
  "refined_end_time": <seconds or null>,
  "description": "<detailed description: tray pickup, boiler placement, rotation direction and count, whether chef left ROI>",
  "reasoning": "<explain why valid or invalid: address each step, stroke counts per direction, and compliance assessment>"
}}
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
    _fh = logging.FileHandler('noodle_rotation_analysis.log', encoding='utf-8', mode='w')
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
    motion_threshold: float = 0.0
    optical_flow_overlay: bool = True
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    roi: Optional[Tuple[int, int, int, int]] = None

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
    """Represents a detected noodle rotation event."""
    start_time: float
    end_time: float
    confidence: float
    description: str
    phase: int  # 1 or 2
    transfer_count: Optional[int] = None
    rotation_strokes_cw: Optional[int] = None
    rotation_strokes_ccw: Optional[int] = None
    rotation_compliant: Optional[bool] = None
    visibility: Optional[str] = None                # "direct" or "indirect"
    is_valid: Optional[bool] = None                 # Set in phase 2
    verified_count: Optional[int] = None
    reading_correction: Optional[str] = None        # Note if count/sequence was corrected
    reasoning: Optional[str] = None
    real_timestamp: Optional[str] = None


class FrameBatch:
    """Manages a batch of frames with size constraints."""

    def __init__(self, batch_id: int, max_size_mb: float, max_frames: int = MAX_FRAMES_PER_BATCH):
        self.batch_id = batch_id
        self.max_size_bytes = (max_size_mb * 0.90) * 1024 * 1024
        self.max_frames = max_frames
        self.frames: List[Tuple[float, str]] = []  # (timestamp, base64_image)
        self.current_size_bytes = 0

    def can_add_frame(self, frame_size: int) -> bool:
        """Check if frame can be added without exceeding size or count limit."""
        if len(self.frames) >= self.max_frames:
            return False
        frame_with_overhead = frame_size + 200
        return (self.current_size_bytes + frame_with_overhead) <= self.max_size_bytes

    def add_frame(self, timestamp: float, base64_frame: str, frame_size: int):
        """Add a frame to the batch."""
        self.frames.append((timestamp, base64_frame))
        self.current_size_bytes += (frame_size + 200)

    def get_size_mb(self) -> float:
        """Get current batch size in MB."""
        return self.current_size_bytes / (1024 * 1024)

    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range covered by this batch."""
        if not self.frames:
            return (0.0, 0.0)
        return (self.frames[0][0], self.frames[-1][0])


# ============================================================================
# 🧠 CORE PIPELINE
# ============================================================================

class NoodleRotationPipeline:
    """Main pipeline for noodle rotation compliance detection and verification."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize OpenAI for Phase 1
        self.openai_client = OpenAI(api_key=config.openai_api_key)

        # Initialize Gemini for Phase 2
        if config.google_api_key:
            self.gemini_client = genai.Client(api_key=config.google_api_key)
        else:
            self.gemini_client = None
            if config.enable_phase2:
                logger.warning("Google API Key not provided. Phase 2 will likely fail.")

        self.context_history: List[str] = []
        self.phase1_detections: List[Detection] = []
        self.phase2_detections: List[Detection] = []
        self.temp_dir = tempfile.mkdtemp(prefix="noodle_rotation_")

        # ROI storage
        self.roi = config.roi
        self.video_fps = None

        # Pre-compute ffmpeg path
        self._ffmpeg_path = shutil.which('ffmpeg')
        if not self._ffmpeg_path:
            for _candidate in [r'C:\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe']:
                if os.path.exists(_candidate):
                    self._ffmpeg_path = _candidate
                    break

        # Pre-compute interpolation flag and target size
        self._interpolation_flag = cv2.INTER_LANCZOS4 if config.image_interpolation.upper() == "LANCZOS" else cv2.INTER_CUBIC
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
        self.token_usage = {
            "phase1_batches": [],
            "phase2_clips": [],
            "total_phase1_tokens": 0,
            "total_phase2_tokens": 0,
            "total_tokens": 0,
            "phase1_raw_responses": [],
            "phase2_raw_responses": [],
        }

        # Create subdirectories in results folder
        self.responses_dir = os.path.join(config.output_dir, "phase2_responses")
        self.frames_dir = os.path.join(config.output_dir, "sample_frames")
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        logger.info(f"Initialized pipeline with temp dir: {self.temp_dir}")
        logger.info(f"Results will be saved to: {config.output_dir}")
        logger.info(f"Analysis Mode: {'Two-Phase (Phase 1 + Phase 2)' if config.enable_phase2 else 'Single-Phase (Phase 1 only)'}")
        logger.info(f"Phase 1 Model: {config.phase1_model_name}")
        if config.enable_phase2:
            logger.info(f"Phase 2 Model: {config.phase2_model_name}")
        logger.info(f"Enable Cropping: {config.enable_cropping}")
        logger.info(f"Rotation Angle: {config.rotation_angle} degrees")
        logger.info(f"Upscale Factor: {config.image_upscale_factor}x")

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp dir: {self.temp_dir}")

    # =========================================================================
    # Frame Processing Methods
    # =========================================================================

    def rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame by the specified rotation angle."""
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

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to the ROI region with surrounding context."""
        if self.roi is None:
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        bw = x2 - x1
        bh = y2 - y1

        cx1 = max(0, x1 - bw // 2)
        cx2 = min(w, x2 + bw // 2)
        cy1 = max(0, y1 - bh)
        cy2 = min(h, y2 + bh)

        return frame[cy1:cy2, cx1:cx2]

    _MAX_UPSCALED_LONG_EDGE = 1920

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale frame by the configured upscale factor and optionally resize to target resolution."""
        if self.config.image_upscale_factor > 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.config.image_upscale_factor)
            new_height = int(height * self.config.image_upscale_factor)
            long_edge = max(new_width, new_height)
            if long_edge > self._MAX_UPSCALED_LONG_EDGE:
                scale = self._MAX_UPSCALED_LONG_EDGE / long_edge
                new_width = int(new_width * scale)
                new_height = int(new_height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=self._interpolation_flag)

        if self._target_size is not None:
            frame = cv2.resize(frame, self._target_size, interpolation=self._interpolation_flag)

        return frame

    def draw_roi_box(self, frame: np.ndarray) -> np.ndarray:
        """Draw a rectangle around the ROI on the full frame."""
        if self.roi is None:
            return frame
        frame_with_box = frame.copy()
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return frame_with_box

    def prepare_frame_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for analysis based on enable_cropping setting."""
        if self.config.enable_cropping:
            cropped = self.crop_frame(frame)
            return self.upscale_frame(cropped)
        else:
            return self.draw_roi_box(frame)

    def _compress_frame(self, frame: np.ndarray, format_override: Optional[str] = None, max_long_edge: Optional[int] = None) -> str:
        """Compress frame and convert to base64 using configured format (PNG or JPEG)."""
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
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])

        return base64.b64encode(bytes(buf)).decode("utf-8")

    # =========================================================================
    # Frame Extraction Methods
    # =========================================================================

    def extract_frames(self, video_path: str) -> List[Tuple[float, str]]:
        """Extract frames from video at specified FPS and convert to base64."""
        logger.info(f"Extracting frames from {video_path} at {self.config.fps} FPS")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        self.video_fps = video_fps
        logger.info(f"Video: {video_fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")

        frames = []
        time_interval = 1.0 / self.config.fps
        next_extract_time = 0.0
        frame_count = 0
        prev_gray_for_flow = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / video_fps

            if current_time >= next_extract_time:
                prepared = self.prepare_frame_for_analysis(frame)
                if self.config.optical_flow_overlay:
                    prepared, prev_gray_for_flow = apply_optical_flow_overlay(prepared, prev_gray_for_flow)
                base64_frame = self._compress_frame(prepared, max_long_edge=self.config.phase1_max_long_edge)
                frames.append((current_time, base64_frame))

                if len(frames) % 50 == 0:
                    logger.info(f"Extracted {len(frames)} frames (timestamp: {current_time:.2f}s)")

                next_extract_time += time_interval

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames total")
        return frames

    def extract_frames_phase2(self, video_path: str) -> List[Tuple[float, str]]:
        """Extract frames for Phase 2 verification — always uses PNG format for best quality."""
        logger.info(f"Extracting Phase 2 frames from {video_path} at {self.config.fps} FPS (PNG format)")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        time_interval = 1.0 / self.config.fps
        next_extract_time = 0.0
        frame_count = 0
        prev_gray_for_flow = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / video_fps

            if current_time >= next_extract_time:
                prepared = self.prepare_frame_for_analysis(frame)
                if self.config.optical_flow_overlay:
                    prepared, prev_gray_for_flow = apply_optical_flow_overlay(prepared, prev_gray_for_flow)
                base64_frame = self._compress_frame(prepared, format_override=self.config.phase2_image_format)
                frames.append((current_time, base64_frame))
                next_extract_time += time_interval

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} Phase 2 frames (PNG format)")
        return frames

    # =========================================================================
    # Batch Management
    # =========================================================================

    def create_batches(self, frames: List[Tuple[float, str]]) -> List[FrameBatch]:
        """Create batches of frames respecting size limits and overlap."""
        logger.info(f"Creating batches from {len(frames)} frames with {self.config.batch_overlap_frames} frames overlap")

        batches = []
        current_batch = FrameBatch(0, self.config.max_batch_size_mb, self.config.max_frames_per_batch)

        for timestamp, base64_frame in frames:
            frame_size = len(base64_frame)

            if not current_batch.can_add_frame(frame_size):
                if current_batch.frames:
                    overlap_frames = []
                    if self.config.batch_overlap_frames > 0:
                        overlap_frames = current_batch.frames[-self.config.batch_overlap_frames:]

                    batches.append(current_batch)
                    logger.info(f"Batch {current_batch.batch_id}: {len(current_batch.frames)} frames, {current_batch.get_size_mb():.2f} MB")

                    current_batch = FrameBatch(len(batches), self.config.max_batch_size_mb, self.config.max_frames_per_batch)

                    for ts, b64 in overlap_frames:
                        current_batch.add_frame(ts, b64, len(b64))

            current_batch.add_frame(timestamp, base64_frame, frame_size)

        if current_batch.frames:
            batches.append(current_batch)
            logger.info(f"Batch {current_batch.batch_id}: {len(current_batch.frames)} frames, {current_batch.get_size_mb():.2f} MB")

        return batches

    # =========================================================================
    # Phase 1 Analysis
    # =========================================================================

    def analyze_batch_phase1(self, batch: FrameBatch, context: str) -> Dict[str, Any]:
        """Analyze a batch in phase 1 using GPT-5-mini."""
        logger.info(f"Analyzing batch {batch.batch_id} (Phase 1) - {len(batch.frames)} frames")

        mime_type = "image/png" if self.config.image_format == "PNG" else "image/jpeg"

        content = [
            {"type": "text", "text": PHASE1_SYSTEM_PROMPT.format(context=context)},
        ]

        for timestamp, base64_frame in batch.frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_frame}"}
            })
            content.append({"type": "text", "text": f"t={timestamp:.1f}"})

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.phase1_model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"}
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
            if not isinstance(result, dict):
                if isinstance(result, list):
                    logger.warning(f"Batch {batch.batch_id}: LLM returned a list instead of an object. Wrapping.")
                    result = {"detections": result, "context_summary": ""}
                else:
                    result = {"detections": [], "context_summary": ""}

            raw_detections = result.get("detections", [])
            logger.info(f"[DEBUG] Batch {batch.batch_id}: Found {len(raw_detections)} raw detections before filtering")
            print(f"[DEBUG] Batch {batch.batch_id}: Found {len(raw_detections)} raw detections before filtering")
            for i, det in enumerate(raw_detections):
                confidence = det.get("confidence", 0) or 0
                transfer_count = det.get("transfer_count", "N/A")
                start_time = det.get("start_time", "N/A")
                end_time = det.get("end_time", "N/A")
                filter_status = "PASSED" if confidence >= self.config.confidence_threshold else f"FILTERED (conf {confidence} < threshold {self.config.confidence_threshold})"
                logger.info(f"  Detection {i+1}: transfer_count={transfer_count}, time={start_time}-{end_time}, conf={confidence}, status={filter_status}")
                print(f"  Detection {i+1}: transfer_count={transfer_count}, time={start_time}-{end_time}, conf={confidence}, status={filter_status}")

            # Track token usage — support both Chat Completions (prompt_tokens/
            # completion_tokens) and Responses API (input_tokens/output_tokens).
            _u = response.usage
            _prompt     = int(getattr(_u, "prompt_tokens", 0) or getattr(_u, "input_tokens", 0) or 0)
            _completion = int(getattr(_u, "completion_tokens", 0) or getattr(_u, "output_tokens", 0) or 0)
            tokens_used = int(getattr(_u, "total_tokens", 0) or 0) or (_prompt + _completion)
            self.token_usage["phase1_batches"].append({
                "batch_id": batch.batch_id,
                "frames": len(batch.frames),
                "prompt_tokens": _prompt,
                "completion_tokens": _completion,
                "total_tokens": tokens_used
            })
            self.token_usage["total_phase1_tokens"] += tokens_used
            self.token_usage["total_tokens"] += tokens_used

            logger.info(f"Batch {batch.batch_id} complete. Tokens: {tokens_used}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing batch {batch.batch_id}: {e}")
            return {"detections": [], "context_summary": ""}

    def run_phase1(self, video_path: str) -> List[Detection]:
        """Run phase 1: Initial noodle rotation detection."""
        logger.info(f"PHASE 1: Scanning {video_path}")
        all_frames = self.extract_frames(video_path)
        self.all_frames = all_frames
        batches = self.create_batches(all_frames)

        detections = []
        context = "This is the first batch of the video."

        for batch in batches:
            # Motion pre-screening: skip idle batches (DyToK arXiv 2512.06866)
            if self.config.motion_threshold > 0:
                score = compute_batch_mafd(batch.frames)
                if score < self.config.motion_threshold:
                    logger.info(f"Batch {batch.batch_id} skipped — MAFD {score:.1f} < threshold {self.config.motion_threshold}")
                    continue
            result = self.analyze_batch_phase1(batch, context)
            for det in result.get("detections", []):
                confidence = det.get("confidence", 0) or 0
                if confidence >= self.config.confidence_threshold:
                    start_time = float(det.get("start_time") or 0.0)
                    end_time = float(det.get("end_time") or 0.0)

                    if end_time <= start_time:
                        logger.warning(f"Skipping invalid detection interval: {start_time:.2f}s - {end_time:.2f}s")
                        continue

                    detections.append(Detection(
                        start_time=start_time,
                        end_time=end_time,
                        confidence=float(confidence),
                        description=det.get("description", "") or "",
                        phase=1,
                        transfer_count=det.get("transfer_count"),
                        rotation_strokes_cw=det.get("rotation_strokes_cw"),
                        rotation_strokes_ccw=det.get("rotation_strokes_ccw"),
                        rotation_compliant=det.get("rotation_compliant"),
                        visibility=det.get("visibility"),
                    ))

            context_summary = result.get("context_summary", "")
            if context_summary:
                self.context_history.append(context_summary)
                context = " | ".join(self.context_history[-1:])

        logger.info(f"Phase 1 complete: {len(detections)} potential intervals")
        self.phase1_detections = detections

        # Save representative frames per detection for dashboard display
        for idx, det in enumerate(detections):
            win_start = max(0.0, det.start_time - self.config.clip_buffer_seconds)
            win_end   = det.end_time + self.config.clip_buffer_seconds
            event_frames = [(ts, b64) for ts, b64 in all_frames if win_start <= ts <= win_end]
            if not event_frames:
                continue
            p1_dir = os.path.join(self.config.output_dir, "phase1_event_frames", f"event_{idx + 1}")
            os.makedirs(p1_dir, exist_ok=True)
            step = max(1, len(event_frames) // 5)
            for fi, (ts, b64) in enumerate(event_frames[::step][:5]):
                img_bytes = base64.b64decode(b64)
                with open(os.path.join(p1_dir, f"frame_{fi + 1}_{ts:.1f}s.png"), "wb") as imgf:
                    imgf.write(img_bytes)

        return detections

    # =========================================================================
    # Video Clipping
    # =========================================================================

    def clip_video_segment(self, video_path: str, start_time: float, end_time: float, output_path: str):
        """Clip a segment from video using ffmpeg."""
        start_with_buffer = max(0, start_time - self.config.clip_buffer_seconds)
        end_with_buffer = end_time + self.config.clip_buffer_seconds
        duration = end_with_buffer - start_with_buffer

        ffmpeg_path = self._ffmpeg_path
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found. Please install FFmpeg to use this pipeline.")

        cmd = [
            ffmpeg_path, '-i', video_path, '-ss', str(start_with_buffer), '-t', str(duration),
            '-c', 'copy', '-y', output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
            logger.info(f"Saved clip: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error clipping video: {e.stderr}")
            raise

    # =========================================================================
    # Phase 2 Analysis
    # =========================================================================

    def analyze_clip_phase2(
        self,
        clip_frames: List[Tuple[float, str]],
        detection_context: str,
        clip_index: int,
        phase1_detection: Detection,
    ) -> Dict[str, Any]:
        """Analyze a clipped segment in phase 2 using Gemini."""
        logger.info(f"Analyzing clip (Phase 2) with Gemini model: {self.config.phase2_model_name}")

        phase1_info = (
            f"Phase 1 detected transfer count: {phase1_detection.transfer_count}\n"
            f"Original video time interval: {phase1_detection.start_time:.2f}s to {phase1_detection.end_time:.2f}s"
        )
        full_context = f"{detection_context}\n{phase1_info}"

        system_instruction = PHASE2_SYSTEM_PROMPT.format(context=full_context)

        content = [
            f"Verify if this clip contains a valid noodle rotation event.\n"
            f"ORIGINAL VIDEO TIME: {phase1_detection.start_time:.2f}s to {phase1_detection.end_time:.2f}s\n"
            f"Phase 1 detected transfer count: {phase1_detection.transfer_count}\n"
            f"{len(clip_frames)} frames provided below."
        ]

        for timestamp, base64_frame in clip_frames:
            try:
                image_data = base64.b64decode(base64_frame)
                image = Image.open(io.BytesIO(image_data))
                original_timestamp = phase1_detection.start_time + timestamp
                content.append(f"clip={timestamp:.1f} video={original_timestamp:.1f}")
                content.append(image)
            except Exception as e:
                logger.warning(f"Failed to process frame at {timestamp}s: {e}")

        try:
            if not self.gemini_client:
                raise ValueError("Gemini client not initialized. Check GOOGLE_API_KEY.")

            response = self.gemini_client.models.generate_content(
                model=self.config.phase2_model_name,
                contents=content,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "system_instruction": system_instruction
                }
            )

            result_text = response.text
            self.token_usage["phase2_raw_responses"].append(
                f"=== Clip {clip_index} ===\n{result_text}"
            )
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "").replace("```", "").strip()

            result = json.loads(result_text)
            if not isinstance(result, dict):
                if isinstance(result, list) and len(result) > 0:
                    logger.warning(f"Clip {clip_index}: Gemini returned a list instead of an object. Using first element.")
                    result = result[0]
                else:
                    result = {"is_valid": False, "confidence": 0.0, "reasoning": "Invalid JSON structure from Gemini"}

            # Track token usage
            tokens_used = 0
            try:
                if hasattr(response, 'usage_metadata'):
                    tokens_used = response.usage_metadata.total_token_count
                    self.token_usage["phase2_clips"].append({
                        "clip_index": clip_index,
                        "frames": len(clip_frames),
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": tokens_used
                    })
                    self.token_usage["total_phase2_tokens"] += tokens_used
                    self.token_usage["total_tokens"] += tokens_used
            except Exception as token_err:
                logger.warning(f"Could not extract token usage from Gemini response: {token_err}")

            return result

        except Exception as e:
            logger.error(f"Error in phase 2 analysis (Gemini): {e}")
            return {"is_valid": False, "confidence": 0.0, "reasoning": f"Gemini Error: {e}"}

    def consolidate_detections(self, detections: List[Detection]) -> List[Detection]:
        """Consolidate overlapping detection intervals."""
        if not detections:
            return []

        sorted_detections = sorted(detections, key=lambda d: d.start_time)
        consolidated = []
        current = sorted_detections[0]

        current_start = max(0, current.start_time - self.config.clip_buffer_seconds)
        current_end = current.end_time + self.config.clip_buffer_seconds

        for detection in sorted_detections[1:]:
            det_start = max(0, detection.start_time - self.config.clip_buffer_seconds)
            det_end = detection.end_time + self.config.clip_buffer_seconds

            if det_start <= current_end + 1:
                current_end = max(current_end, det_end)
                current.confidence = max(current.confidence, detection.confidence)
                if detection.description not in current.description:
                    current.description += f" | {detection.description}"
                # Accumulate transfer count across merged detections
                if detection.transfer_count and current.transfer_count is not None:
                    current.transfer_count += detection.transfer_count
                elif detection.transfer_count:
                    current.transfer_count = detection.transfer_count
                
                # Accumulate rotation strokes across merged detections
                if detection.rotation_strokes_cw and current.rotation_strokes_cw is not None:
                    current.rotation_strokes_cw += detection.rotation_strokes_cw
                elif detection.rotation_strokes_cw:
                    current.rotation_strokes_cw = detection.rotation_strokes_cw
                    
                if detection.rotation_strokes_ccw and current.rotation_strokes_ccw is not None:
                    current.rotation_strokes_ccw += detection.rotation_strokes_ccw
                elif detection.rotation_strokes_ccw:
                    current.rotation_strokes_ccw = detection.rotation_strokes_ccw
            else:
                current.start_time = current_start + self.config.clip_buffer_seconds
                current.end_time = current_end - self.config.clip_buffer_seconds
                consolidated.append(current)
                current = detection
                current_start = det_start
                current_end = det_end

        current.start_time = current_start + self.config.clip_buffer_seconds
        current.end_time = current_end - self.config.clip_buffer_seconds
        consolidated.append(current)

        logger.info(f"Consolidation: {len(detections)} -> {len(consolidated)} events")
        return consolidated

    def run_phase2(self, video_path: str, phase1_detections: List[Detection]) -> List[Detection]:
        """Run phase 2: Verification using Phase 1 cached frames."""
        logger.info("PHASE 2: Verification")
        consolidated = self.consolidate_detections(phase1_detections)
        logger.info(f"Consolidated {len(phase1_detections)} Phase 1 detections into {len(consolidated)} events for verification")

        # Reuse Phase 1 frames (already extracted and pre-processed)
        source_frames: List[Tuple[float, str]] = getattr(self, "all_frames", None) or []
        if not source_frames:
            logger.warning("Phase 1 frames not cached; re-extracting for Phase 2")
            source_frames = self.extract_frames_phase2(video_path)

        verified = []

        for idx, detection in enumerate(consolidated):
            logger.info(f"Verifying {idx+1}/{len(consolidated)}: {detection.start_time:.1f}s-{detection.end_time:.1f}s")

            buf = self.config.clip_buffer_seconds
            win_start = max(0.0, detection.start_time - buf)
            win_end = detection.end_time + buf
            clip_frames = [
                (ts - detection.start_time, b64)
                for ts, b64 in source_frames
                if win_start <= ts <= win_end
            ]
            logger.info(f"  Using {len(clip_frames)} frames from [{win_start:.1f}s, {win_end:.1f}s] window")

            # Save verification frames for dashboard display
            if clip_frames:
                verify_dir = os.path.join(
                    self.config.output_dir, "phase2_frames", f"event_{idx + 1}"
                )
                os.makedirs(verify_dir, exist_ok=True)
                step = max(1, len(clip_frames) // 5)
                for fi, (rel_ts, b64) in enumerate(clip_frames[::step][:5]):
                    img_bytes = base64.b64decode(b64)
                    fname = f"frame_{fi + 1}_{rel_ts + detection.start_time:.1f}s.png"
                    with open(os.path.join(verify_dir, fname), "wb") as imgf:
                        imgf.write(img_bytes)

            context = f"Potential noodle rotation event: {detection.description}"
            result = self.analyze_clip_phase2(clip_frames, context, idx + 1, detection)

            # Save response
            with open(os.path.join(self.responses_dir, f"response_{idx+1}.json"), 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            if result.get("is_valid", False):
                verified_count = result.get("verified_count") or detection.transfer_count

                # Detect if count was corrected
                reading_correction = None
                if (verified_count is not None and detection.transfer_count is not None
                        and verified_count != detection.transfer_count):
                    reading_correction = (
                        f"Count corrected from {detection.transfer_count} to {verified_count}"
                    )

                verified_det = Detection(
                    start_time=detection.start_time,
                    end_time=detection.end_time,
                    confidence=float(result.get("confidence", 0)),
                    description=result.get("description", ""),
                    phase=2,
                    transfer_count=verified_count,
                    rotation_strokes_cw=result.get("rotation_strokes_cw", detection.rotation_strokes_cw),
                    rotation_strokes_ccw=result.get("rotation_strokes_ccw", detection.rotation_strokes_ccw),
                    rotation_compliant=result.get("rotation_compliant", detection.rotation_compliant),
                    visibility=result.get("visibility"),
                    is_valid=True,
                    verified_count=verified_count,
                    reading_correction=reading_correction,
                    reasoning=result.get("reasoning", ""),
                )
                verified.append(verified_det)
                logger.info(
                    f"[OK] VERIFIED: transfer_count={verified_count}, "
                    f"time={detection.start_time:.1f}s-{detection.end_time:.1f}s"
                )
            else:
                logger.info(f"[X] REJECTED: {result.get('reasoning')}")

        self.phase2_detections = verified
        return verified

    # =========================================================================
    # Results & Output
    # =========================================================================

    def create_merged_video(self, video_path: str):
        """Create a merged video of all verified clips."""
        if not self.phase2_detections:
            return

        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for idx, detection in enumerate(self.phase2_detections, 1):
                clip_temp = os.path.join(self.temp_dir, f"merge_clip_{idx}.mp4")
                self.clip_video_segment(video_path, detection.start_time, detection.end_time, clip_temp)
                f.write(f"file '{clip_temp.replace(os.sep, '/')}'\n")

        merged_path = os.path.join(self.config.output_dir, "merged_verified_clips.mp4")
        ffmpeg_path = self._ffmpeg_path

        if ffmpeg_path and os.path.exists(ffmpeg_path):
            try:
                subprocess.run([
                    ffmpeg_path, '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-c', 'copy', '-y', merged_path
                ], check=True, capture_output=True)
                logger.info(f"Merged video created: {merged_path}")
            except Exception as e:
                logger.error(f"Failed to create merged video: {e}")

    def save_results(self):
        """Save all analysis results."""
        output_path = os.path.join(self.config.output_dir, "analysis_results.json")
        results = {
            "metadata": {
                "video": self.config.input_video_path,
                "timestamp": datetime.now().isoformat(),
                "fps": self.config.fps,
                "video_fps": self.video_fps,
                "roi": self.roi,
                "rotation_angle": self.config.rotation_angle,
                "enable_cropping": self.config.enable_cropping,
                "upscale_factor": self.config.image_upscale_factor,
                "phase1_model": self.config.phase1_model_name,
                "phase2_model": self.config.phase2_model_name if self.config.enable_phase2 else None
            },
            "phase1": [asdict(d) for d in self.phase1_detections],
            "phase2": [asdict(d) for d in self.phase2_detections],
            "stats": {
                "p1_count": len(self.phase1_detections),
                "p2_count": len(self.phase2_detections)
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # Save token usage
        token_output_path = os.path.join(self.config.output_dir, "token_usage.json")
        with open(token_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_usage, f, indent=2)
        logger.info(f"Token usage saved to: {token_output_path}")
        logger.info(
            f"Total tokens: {self.token_usage['total_tokens']} "
            f"(Phase1: {self.token_usage['total_phase1_tokens']}, Phase2: {self.token_usage['total_phase2_tokens']})"
        )

        # Text summary
        summary_path = os.path.join(self.config.output_dir, "analysis_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"NOODLE ROTATION COMPLIANCE ANALYSIS SUMMARY\n{'='*50}\n")
            f.write(f"Video: {self.config.input_video_path}\n")
            f.write(f"Analysis FPS: {self.config.fps}\n")
            f.write(f"Video FPS: {self.video_fps:.2f}\n")
            f.write(f"ROI: {self.roi}\n")
            f.write(f"Rotation: {self.config.rotation_angle} degrees\n")
            f.write(f"Cropping: {'Enabled' if self.config.enable_cropping else 'Disabled (Box overlay)'}\n")
            f.write(f"Phase 1 Model: {self.config.phase1_model_name}\n")
            if self.config.enable_phase2:
                f.write(f"Phase 2 Model: {self.config.phase2_model_name}\n")
            f.write(f"\nPhase 1 Detections: {len(self.phase1_detections)}\n")
            f.write(f"Phase 2 Verified: {len(self.phase2_detections)}\n\n")
            f.write(f"Total Tokens: {self.token_usage['total_tokens']}\n")
            f.write(f"  - Phase 1: {self.token_usage['total_phase1_tokens']}\n")
            f.write(f"  - Phase 2: {self.token_usage['total_phase2_tokens']}\n\n")
            f.write("-" * 50 + "\n")
            f.write("VERIFIED NOODLE ROTATION EVENTS:\n")
            f.write("-" * 50 + "\n")

            detections = self.phase2_detections if self.config.enable_phase2 else self.phase1_detections
            for idx, d in enumerate(detections, 1):
                f.write(f"\n{idx}. Time: {d.start_time:.1f}s - {d.end_time:.1f}s\n")
                f.write(f"   Transfer Count: {d.transfer_count}\n")
                f.write(f"   Visibility: {d.visibility}\n")
                f.write(f"   Confidence: {d.confidence:.2f}\n")
                f.write(f"   Description: {d.description}\n")
                if d.reading_correction:
                    f.write(f"   Correction: {d.reading_correction}\n")

        logger.info(f"Summary saved to: {summary_path}")

    # =========================================================================
    # Main Run Method
    # =========================================================================

    def run(self):
        """Main pipeline execution."""
        try:
            print("\n" + "="*60)
            print("NOODLE ROTATION COMPLIANCE PIPELINE")
            print("="*60)
            print(f"Video: {self.config.input_video_path}")
            print(f"Phase 1 Model: {self.config.phase1_model_name}")
            if self.config.enable_phase2:
                print(f"Phase 2 Model: {self.config.phase2_model_name}")
            print("="*60)

            # Phase 1
            phase1_detections = self.run_phase1(self.config.input_video_path)

            # Phase 2
            if self.config.enable_phase2:
                if phase1_detections:
                    self.run_phase2(self.config.input_video_path, phase1_detections)
                    self.create_merged_video(self.config.input_video_path)
                else:
                    logger.info("No Phase 1 detections found. Skipping Phase 2.")
                    self.phase2_detections = []
            else:
                logger.info("Single-phase mode: Skipping Phase 2.")
                self.phase2_detections = phase1_detections

            self.save_results()

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


# ============================================================================
# 🚀 ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Noodle Rotation Compliance Analysis")
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    video_path = args.video_path or DEFAULT_INPUT_VIDEO
    if not video_path:
        print("Error: No video path provided.")
        print("Usage: python noodle_rotation_compliance.py <video_path>")
        print("OR set INPUT_VIDEO_PATH in .env file")
        return

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY is not set in .env!")
        return
    if not GOOGLE_API_KEY and ENABLE_PHASE2:
        print("[WARNING] GOOGLE_API_KEY is not set. Phase 2 will fail.")

    print("="*60)
    print(f"  - Analysis Mode: {'Two-Phase' if ENABLE_PHASE2 else 'Single-Phase'}")
    print(f"  - Phase 1 Model: {PHASE1_MODEL_NAME}")
    if ENABLE_PHASE2:
        print(f"  - Phase 2 Model: {PHASE2_MODEL_NAME}")
    print("="*60)

    output_dir = os.path.join(args.output, Path(video_path).stem)
    config = PipelineConfig(input_video_path=video_path, output_dir=output_dir)
    pipeline = NoodleRotationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
