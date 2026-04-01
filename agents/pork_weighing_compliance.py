"""
Pork Weighing Compliance Analysis Agent
-----------------------------------------
Detects pork weighing events and extracts scale readings from video footage.
Uses a two-phase pipeline:
  - Phase 1: GPT-5-mini for event detection and initial OCR
  - Phase 2: Gemini 2.5 Pro for verification and reading confirmation

This agent wraps pork_weighing_analysis.py (parent directory) for web deployment.
See: ../pork_weighing_analysis.py
"""

# The full implementation lives in:
#   webapp/pipeline_adapter.py  — headless runner
#   webapp/worker.py            — background task processor
#   webapp/main.py              — FastAPI endpoints

"""
Pork Weighing Analysis Pipeline with Two-Phase Detection
---------------------------------------------------------
A unified pipeline to detect pork weighing events and extract scale readings.

Phase 1: GPT-5-mini - Detect pork weighing events and initial OCR readings
Phase 2: Gemini-2.5-pro - Verify events and confirm scale readings

Features:
- Interactive ROI selection for weighing scale display
- Frame cropping, rotation, and upscaling for better OCR
- Two-phase analysis with event detection + reading verification
"""

import sys
import os
import io
import json
import base64
import logging
import shutil
import tempfile
import subprocess
import traceback
import argparse
import time
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
    import pandas as pd
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install requirements: pip install openai opencv-python-headless numpy pillow google-generativeai python-dotenv pandas")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# ⚙️ AGENT-LEVEL PARAMETER DEFAULTS
# Canonical defaults for the pork weighing task.
# webapp/config.py imports these — the agent file is the single source of truth.
# Individual parameters can be overridden via environment variables EXCEPT where
# marked "fixed" (those are non-negotiable for correct task behaviour).
# ============================================================================

AGENT_PHASE1_MODEL_NAME       = "gpt-5-mini"
AGENT_PHASE2_MODEL_NAME       = "gemini-2.5-pro"
AGENT_FPS                     = 1.0
AGENT_CONFIDENCE_THRESHOLD    = 0.1
AGENT_MAX_BATCH_SIZE_MB       = 35.0
AGENT_CLIP_BUFFER_SECONDS     = 2
AGENT_MAX_FRAMES_PER_BATCH    = 300
AGENT_BATCH_OVERLAP_FRAMES    = 2
AGENT_IMAGE_QUALITY           = 90
AGENT_IMAGE_UPSCALE_FACTOR    = 1.0
AGENT_IMAGE_TARGET_RESOLUTION = "auto"
AGENT_IMAGE_FORMAT            = "JPEG"
AGENT_PHASE2_IMAGE_FORMAT     = "PNG"
AGENT_IMAGE_INTERPOLATION     = "CUBIC"
AGENT_ENABLE_CROPPING         = True
AGENT_ROTATION_ANGLE          = 270

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
IMAGE_QUALITY           = int(os.getenv("IMAGE_QUALITY",         str(AGENT_IMAGE_QUALITY)))
IMAGE_UPSCALE_FACTOR    = float(os.getenv("IMAGE_UPSCALE_FACTOR", str(AGENT_IMAGE_UPSCALE_FACTOR)))
IMAGE_TARGET_RESOLUTION = os.getenv("IMAGE_TARGET_RESOLUTION",   AGENT_IMAGE_TARGET_RESOLUTION)
IMAGE_FORMAT            = os.getenv("IMAGE_FORMAT", AGENT_IMAGE_FORMAT).upper()
PHASE2_IMAGE_FORMAT     = AGENT_PHASE2_IMAGE_FORMAT # always "PNG" — not env-overridable for this task
IMAGE_INTERPOLATION     = os.getenv("IMAGE_INTERPOLATION", AGENT_IMAGE_INTERPOLATION).upper()
# Cropping / Rotation
ENABLE_CROPPING = os.getenv("ENABLE_CROPPING", str(AGENT_ENABLE_CROPPING)).lower() == "true"
ROTATION_ANGLE  = int(os.getenv("ROTATION_ANGLE", str(AGENT_ROTATION_ANGLE)))

# Input/Output Defaults
DEFAULT_INPUT_VIDEO = os.getenv("INPUT_VIDEO_PATH", "")
DEFAULT_OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "./results_pork_weighing")

# ============================================================================
# 🤖 PROMPTS
# ============================================================================

PHASE1_SYSTEM_PROMPT = """You are a video analysis assistant specialized in monitoring weighing scale operations in a kitchen. Your task is to analyze video frames and identify when the chef is weighing PORK on the scale, and accurately READ the digital display.

FRAME CONTEXT:
- Each frame is a CROPPED AND UPSCALED ROI centred on the weighing scale; the display and platform fill most of the image
- If the digital display is unreadable it is due to glare, obstruction, or angle — NOT distance from camera

SCALE IDENTIFICATION:
- This ROI captures ONE specific scale; call it scale "1" unless prior batch context specifies otherwise
- Focus on the display unit, the weighing platform/bowl area, and any items on it

IMPORTANT: A weighing event is COMPLETE only when the pork/bowl has been REMOVED from the scale.
There may be cases when pork is already present in the bowl when the chef places it on the scale. Conclude the event only when the pork/bowl from the respective scale is removed.
Wait for the FINAL STABLE READING before the pork is removed - this is the reading to capture.

EVENT DETECTION — BROAD CRITERIA (use all three signals together):
A weighing event should be flagged if ANY of the following combinations is observed across the frames in this batch:
1. Pork or a bowl containing meat is visibly present on or near the scale platform (even partially visible inside the ROI), AND the scale digital display shows a non-zero / changed reading compared to an empty-scale state.
2. A chef actively places, adjusts, or removes a bowl/tray with meat on the scale platform — the standard placement-and-removal workflow.
3. Pork/meat is visible inside the scale ROI, the scale reading has clearly changed (indicating load), AND later frames show the bowl being removed from the platform — this FULL SEQUENCE (presence + reading change + removal) constitutes a positive weighing event even if the exact placement moment was not captured.

TIMESTAMP READING:
- Frame timestamps are provided with each image
- Record the exact timestamp when readings are detected
- The END TIME should be when the pork/bowl is being removed or just after

WHAT TO DETECT - Pork Weighing Events:
1. Visual signs of PORK or a bowl/tray with meat visible on or inside the scale platform ROI
2. Any interaction with the scale platform — placing, adjusting, or removing items
3. Scale reading that is non-zero / different from an empty baseline, indicating something is being weighed
4. Which scale number (1, 2, etc.) the event is occurring on
NOTE: A readable digital display is NOT required to report a detection. Pork visibility + scale reading change + bowl removal is sufficient to flag an event.

DETECTION RULE — ALWAYS REPORT, ADJUST CONFIDENCE:
- If you observe pork/meat inside the scale ROI AND the scale reading appears non-zero or changed, ALWAYS create a detection entry
- If a bowl with pork was present and is subsequently removed, flag the moment of removal as the end_time
- The digital display may be partially cut off, obscured, or outside the frame — this does NOT prevent reporting
- A readable display → set scale_reading + appropriate confidence
- Display not visible but pork+removal sequence observed → set scale_reading to null, confidence 0.1–0.2, reading_state "obscured"
- Do NOT skip a detection just because the display is unreadable — Phase 2 will handle verification
- "No detections" should be returned ONLY when there is genuinely no scale interaction AND no pork visible on the scale platform in the entire interval

READING EXTRACTION REQUIREMENTS:
- Read the EXACT numerical value shown on the digital display
- Note the UNIT of measurement (grams, kg, lbs, oz, etc.) if visible
- Capture the FINAL STABLE READING before pork removal
- Reference standard weights are 60g (regular portion) and 120g (large portion), usually the weights on the scale would be around these values.
- Report the exact displayed reading even if it deviates from standards
- If digits are not readable, set scale_reading to null and confidence below 0.4

CONFIDENCE: 0.8-1.0=HIGH(all digits clear) | 0.6-0.7=MED-HIGH(minor blur) | 0.4-0.5=MED(some digits unclear) | 0.2-0.3=LOW(hard to read) | 0.0-0.1=UNREADABLE(event visible, digits not)

Context from previous batch: {context}

Analyze the provided frames and identify ALL pork weighing events with their corresponding scale readings. For each detection, provide:
- start_time: Timestamp when weighing begins
- end_time: Timestamp when weighing ends (pork being removed)
- scale: Scale number as a string — use "1" for the scale visible in this ROI crop (only use a different number if the context from a prior batch specifies one)
- scale_reading: The numerical value displayed on the scale (or null if unreadable)
- unit: Unit of measurement (if visible)
- reading_state: "stable", "fluctuating", or "obscured"
- confidence: Your confidence in the reading accuracy (0.0 to 1.0)
- description: Brief description of what was observed

Return your response in JSON format:
{{
  "detections": [
    {{
      "start_time": <timestamp as float>,
      "end_time": <timestamp as float>,
      "scale": "<scale number as string, e.g. '1' or '2'>",
      "scale_reading": <number or null>,
      "unit": "<unit or null>",
      "reading_state": "<stable/fluctuating/obscured>",
      "confidence": <0.0-1.0>,
      "description": "<brief description of the weighing event>"
    }}
  ],
  "context_summary": "<summary of important context for next batch, including any ongoing weighing activity>"
}}
"""

PHASE2_SYSTEM_PROMPT = """You are a video analysis assistant performing detailed verification of detected pork weighing events and scale readings. You are analyzing frames from a video clip that was flagged as potentially containing a pork weighing activity.

Your task is to verify if this is a TRUE POSITIVE or FALSE POSITIVE and provide ACCURATE READING VERIFICATION.

IMPORTANT - TIME INTERVALS:
- The time interval for this event has already been determined in Phase 1 analysis
- You should focus on VERIFYING the detection and READING accuracy, not on adjusting timestamps
- The original video timestamps are provided in the context

FRAME ANALYSIS:
- These frames are CROPPED AND UPSCALED to the ROI bounding box — the scale machine fills most of the image
- The scale is NOT small or distant; the crop is centred on it so the display and platform are both prominent
- Analyze the digital readout directly — it should be clearly visible unless physically obstructed or obscured by glare
- Confirm the scale number from Phase 1 context (default "1" unless context says otherwise)

STEP 1 — RE-READ THE FINAL SCALE DISPLAY:
Before anything else, identify the frames toward the END of the clip (just before the pork/bowl is removed from the platform). These represent the stable, settled reading — the most reliable measurement.
- Carefully re-read every digit on the digital display in those final frames
- If the display is not visible in the final frames, use the clearest readable frame anywhere in the clip
- Record this as your `verified_reading`

STEP 2 — RANGE VALIDATION (apply AFTER re-reading):
Standard pork portion weights are 60g (regular) and 120g (large), tolerance ±8g.
Valid ranges: 52–68g OR 112–128g.
- If your re-read value falls OUTSIDE BOTH ranges (i.e. not in 52–68 and not in 112–128), mark `is_valid = false` (FALSE POSITIVE due to out-of-range reading)
- Exception: if the display was completely unreadable in ALL frames (full obstruction/glare across the entire clip), do NOT apply the range check — instead set `verified_reading = null` and keep `is_valid` based on the pork+removal sequence alone

VERIFICATION CRITERIA:
1. Confirm that PORK (not other ingredients) is being weighed — pork visible inside the ROI crop is sufficient
2. Re-read the final display reading (STEP 1 above) and apply range validation (STEP 2 above)
3. Look for the FULL WEIGHING SEQUENCE: pork present in ROI → non-zero scale reading → bowl/pork removal
4. Assess the clarity and readability of the best available display frame

READING VERIFICATION REQUIREMENTS:
- Re-read the digital display value with maximum precision from the END of the clip (final stable frames)
- Verify the unit of measurement
- Check each digit carefully for accuracy
- Report the exact displayed reading in `verified_reading`

CONFIDENCE (be strict): NEVER use 1.0 | 0.8-0.9=HIGH(all digits clear, stable) | 0.6-0.7=MED-HIGH(minor blur) | 0.4-0.5=MED(partial obstruction) | 0.2-0.3=LOW(heavy blur or inferred from Phase 1 with bowl-removal) | 0.0-0.1=UNREADABLE(no readable frame exists)

CRITICAL RULES:
- AUTOMATIC FALSE POSITIVE if the ingredient is clearly NOT pork (e.g. vegetables, sauce, other non-meat items)
- AUTOMATIC FALSE POSITIVE if the re-read value is outside 52–68g AND outside 112–128g (see STEP 2)
- DO NOT mark as false positive solely because the display is not visible in the current frames — the stable reading may have occurred earlier in the clip
- A TRUE POSITIVE requires: (a) pork visible inside ROI + (b) re-read value within a valid range (52–68 or 112–128) + (c) bowl/pork subsequently removed from platform
- If reading is completely obscured across ALL frames, do not apply the range check — set confidence below 0.3 and keep is_valid based on pork+removal sequence
- Provide refined reading if Phase 1 reading appears incorrect
- Lower confidence if there's any uncertainty about reading accuracy
- Use the scale number from Phase 1 context (default "1")

Previous detection context: {context}

Analyze the provided frames carefully and verify the pork weighing event and scale reading accuracy.

Return your response in JSON format:
{{
  \"is_valid\": <true/false>,
  \"scale\": \"<scale number as string, e.g. '1' or '2'>\",
  \"verified_reading\": <number or null>,
  \"unit\": \"<unit or null>\",
  \"reading_state\": \"<stable/fluctuating/obscured or null>\",
  \"confidence\": <0.0-1.0>,
  \"display_visibility\": \"<clear/partial/obscured>\",
  \"ingredient_confirmed\": \"<pork/not_pork/uncertain>\",
  \"reading_correction\": \"<note if Phase 1 reading was corrected, or null>\",
  \"description\": \"<detailed description of the weighing event and reading>\",
  \"reasoning\": \"<explanation of verification including: why valid/invalid, reading accuracy assessment, visibility factors, and confidence justification>\"
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
    _fh = logging.FileHandler('pork_weighing_analysis.log', encoding='utf-8', mode='w')
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
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class Detection:
    """Represents a detected weighing event."""
    start_time: float
    end_time: float
    confidence: float
    description: str
    phase: int  # 1 or 2
    scale: Optional[str] = None  # "1", "2", … numbered left-to-right in frame
    scale_reading: Optional[float] = None
    unit: Optional[str] = None
    reading_state: Optional[str] = None  # "stable", "fluctuating", "obscured"
    is_valid: Optional[bool] = None  # Set in phase 2
    reasoning: Optional[str] = None  # Set in phase 2
    verified_reading: Optional[float] = None  # Phase 2 verified reading
    reading_correction: Optional[str] = None  # Note if reading was corrected


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
# 🎯 ROI SELECTION TOOL
# ============================================================================

class ROISelector:
    """Interactive ROI selector using OpenCV."""
    
    def __init__(self, frame: np.ndarray, window_name: str = "Select Region of Interest"):
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        self.window_name = window_name
        self.roi = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.confirmed = False
        
    def mouse_callback(self, event, x, y, *_):
        """Handle mouse events for rectangle drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.frame = self.original_frame.copy()
                cv2.rectangle(self.frame, self.start_point, self.end_point, (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.frame = self.original_frame.copy()
            cv2.rectangle(self.frame, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # Calculate ROI coordinates (ensure proper order)
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            
            self.roi = (x1, y1, x2, y2)
    
    def select(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Display the frame and allow user to select ROI.
        Returns: (x1, y1, x2, y2) coordinates or None if cancelled.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("ROI SELECTION INSTRUCTIONS")
        print("="*60)
        print("1. Draw a rectangle around the weighing scale display")
        print("2. Click and drag to draw the rectangle")
        print("3. Press 'c' or ENTER to CONFIRM the selection")
        print("4. Press 'r' to RESET and redraw")
        print("5. Press 'q' or ESC to QUIT without selection")
        print("="*60 + "\n")
        
        while True:
            display_frame = self.frame.copy()
            cv2.putText(display_frame, "Draw rectangle around scale display | C: Confirm | R: Reset | Q: Quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.roi:
                x1, y1, x2, y2 = self.roi
                roi_text = f"ROI: ({x1}, {y1}) to ({x2}, {y2}) | Size: {x2-x1}x{y2-y1}"
                cv2.putText(display_frame, roi_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == 13:  # 'c' or Enter to confirm
                if self.roi:
                    self.confirmed = True
                    break
                else:
                    print("Please draw a rectangle first!")
                    
            elif key == ord('r'):  # 'r' to reset
                self.frame = self.original_frame.copy()
                self.roi = None
                self.start_point = None
                self.end_point = None
                print("Selection reset. Draw again.")
                
            elif key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
        
        cv2.destroyAllWindows()
        
        if self.confirmed and self.roi:
            logger.info(f"ROI selected: {self.roi}")
            return self.roi
        else:
            logger.info("ROI selection cancelled")
            return None


# ============================================================================
# 🧠 CORE PIPELINE
# ============================================================================

class PorkWeighingPipeline:
    """Main pipeline for pork weighing event detection and OCR."""
    
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
        self.temp_dir = tempfile.mkdtemp(prefix="pork_weighing_")
        
        # ROI storage
        self.roi = config.roi
        self.video_fps = None

        # Pre-compute ffmpeg path (avoid repeated filesystem searches per Phase 2 clip)
        self._ffmpeg_path = shutil.which('ffmpeg')
        if not self._ffmpeg_path:
            for _candidate in [r'C:\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe']:
                if os.path.exists(_candidate):
                    self._ffmpeg_path = _candidate
                    break

        # Pre-compute interpolation flag and target size (evaluated per frame in upscale_frame)
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
        logger.info(f"Image Format: {config.image_format}")
        logger.info(f"Interpolation: {config.image_interpolation}")

        # Pre-create CLAHE object to avoid repeated allocation in apply_clahe
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp dir: {self.temp_dir}")

    # =========================================================================
    # Frame Processing Methods
    # =========================================================================

    def get_frame_for_roi_selection(self, frame_index: int = 5) -> Tuple[np.ndarray, float]:
        """Get a specific frame from video for ROI selection."""
        cap = cv2.VideoCapture(self.config.input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.config.input_video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.video_fps = video_fps
        
        if frame_index >= total_frames:
            frame_index = min(5, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read frame {frame_index} from video")
        
        logger.info(f"Retrieved frame {frame_index} for ROI selection (Video FPS: {video_fps:.2f})")
        return frame, video_fps

    def select_roi(self) -> bool:
        """Display frame and let user select ROI."""
        frame, fps = self.get_frame_for_roi_selection(frame_index=4)
        
        # Apply rotation if specified
        frame = self.rotate_frame(frame)
        
        # Create ROI selector and get selection
        selector = ROISelector(frame, "Select Weighing Scale Display Region")
        roi = selector.select()
        
        if roi:
            self.roi = roi
            self.config.roi = roi
            logger.info(f"ROI set to: {roi}")
            
            # Save sample frame showing ROI
            frame_with_roi = frame.copy()
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 255, 0), 3)
            sample_path = os.path.join(self.frames_dir, "roi_selection.jpg")
            cv2.imwrite(sample_path, frame_with_roi)
            logger.info(f"Saved ROI selection frame to: {sample_path}")
            
            return True
        else:
            logger.warning("No ROI selected")
            return False

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

    # -------------------------------------------------------------------------
    # Automatic rotation detection
    # -------------------------------------------------------------------------

    _ROTATION_DETECT_PROMPT = (
        "You are inspecting a single CCTV frame from a kitchen weighing station. "
        "A digital weighing scale display is somewhere in the image. "
        "Is the scale display oriented UPRIGHT and READABLE (the digital scale is at the bottom of the weighing scale/machine)? "
        "Reply with a JSON object only, no markdown, no extra text: "
        '{"upright": true} or {"upright": false}'
    )

    def detect_rotation(self, client: "OpenAI") -> int:  # noqa: F821
        """Probe candidate rotation angles and return the one that places the
        scale display upright, as judged by GPT-4o-mini.

        Tries angles in the order [270, 0, 90, 180] (270 is the most common
        orientation for this camera setup). Stops at the first angle that
        receives {"upright": true}.  Falls back to 270 if none passes or if
        the video cannot be read.

        Returns the detected rotation angle (int).
        """
        _PROBE_CANDIDATES = [270, 0, 90, 180]
        _PROBE_FRAME_INDICES = [5, 15, 30]   # sample a few frames for robustness
        _ROTATE_MAP = {
            0:   lambda f: f,
            90:  lambda f: cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE),
            180: lambda f: cv2.rotate(f, cv2.ROTATE_180),
            270: lambda f: cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }

        logger.info("Auto-detecting rotation angle for scale display...")

        cap = cv2.VideoCapture(self.config.input_video_path)
        if not cap.isOpened():
            logger.warning("detect_rotation: cannot open video — keeping current angle")
            return self.config.rotation_angle

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Clamp probe indices to actual video length
        probe_indices = [min(idx, total_frames - 1) for idx in _PROBE_FRAME_INDICES if idx < total_frames]
        if not probe_indices:
            probe_indices = [0]

        # Read the probe frames once (raw, unrotated)
        raw_frames: List[np.ndarray] = []
        for idx in probe_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                raw_frames.append(frame)
        cap.release()

        if not raw_frames:
            logger.warning("detect_rotation: no frames extracted — keeping current angle")
            return self.config.rotation_angle

        # Use first probe frame only (cheapest — one API call per candidate angle)
        probe_frame = raw_frames[0]

        for angle in _PROBE_CANDIDATES:
            rotated = _ROTATE_MAP[angle](probe_frame)

            # Downscale to 512px long-edge to keep token cost low
            h, w = rotated.shape[:2]
            long_edge = max(h, w)
            if long_edge > 512:
                scale = 512 / long_edge
                rotated = cv2.resize(rotated, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)

            # Encode to JPEG base64
            ok, buf = cv2.imencode(".jpg", rotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=20,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                            {"type": "text", "text": self._ROTATION_DETECT_PROMPT},
                        ],
                    }],
                )
                raw = resp.choices[0].message.content.strip()
                # Strip markdown fences if model adds them
                if raw.startswith("```"):
                    raw = raw.split("```")[1].lstrip("json").strip()
                result = json.loads(raw)
                if result.get("upright") is True:
                    logger.info(f"detect_rotation: angle {angle}° confirmed upright by model")
                    return angle
                else:
                    logger.info(f"detect_rotation: angle {angle}° → not upright, trying next")
            except Exception as exc:
                logger.warning(f"detect_rotation: model call failed for {angle}°: {exc}")

        logger.warning("detect_rotation: no angle confirmed upright — defaulting to 270°")
        return 270

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to ROI."""
        if self.roi is None:
            return frame

        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2]

    # Agent uses a padded crop (wider than the tight ROI in final.py), so the
    # upscaled result can exceed available memory. Cap the long edge to 1280px —
    # equivalent to what final.py produces from a typical tight ROI after 2×
    # upscale — to keep frame sizes consistent between the two implementations.
    _MAX_UPSCALED_LONG_EDGE = 1280

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

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel (LAB space) to enhance scale display contrast."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Use the cached CLAHE object instead of creating a new one per frame
        enhanced_l = self._clahe.apply(l)
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def prepare_frame_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for OCR/Analysis based on enable_cropping setting."""
        if self.config.enable_cropping:
            cropped = self.crop_frame(frame)
            upscaled = self.upscale_frame(cropped)
            return self.apply_clahe(upscaled)
        else:
            return self.draw_roi_box(frame)

    def save_clahe_preview_frames(self, video_path: str) -> None:
        """Extract 7–10 evenly spaced CLAHE-enhanced frames and save for dashboard preview.

        Runs regardless of whether any weighing events are detected.
        Frames are saved to output_dir/clahe_preview_frames/.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"save_clahe_preview_frames: cannot open {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps if video_fps > 0 else 0

        n_frames = min(10, max(7, total_frames // max(1, int(video_fps * 30))))
        n_frames = max(7, min(10, n_frames))

        out_dir = os.path.join(self.config.output_dir, "clahe_preview_frames")
        os.makedirs(out_dir, exist_ok=True)

        positions = [int(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]

        saved = 0
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
            timestamp = pos / video_fps if video_fps > 0 else 0
            rotated = self.rotate_frame(frame)
            if self.config.enable_cropping and self.roi:
                processed = self.apply_clahe(self.upscale_frame(self.crop_frame(rotated)))
            else:
                processed = self.apply_clahe(self.upscale_frame(rotated))
            fname = f"frame_{saved + 1:02d}_{timestamp:.1f}s.png"
            cv2.imwrite(os.path.join(out_dir, fname), processed)
            saved += 1

        cap.release()
        logger.info(f"Saved {saved} CLAHE preview frames to {out_dir}")

    def _compress_frame(self, frame: np.ndarray, format_override: Optional[str] = None) -> str:
        """Compress frame and convert to base64 using configured format (PNG or JPEG)."""
        image_format = format_override.upper() if format_override else self.config.image_format

        if image_format == "PNG":
            _, buf = cv2.imencode(".png", frame)
        else:
            quality = getattr(self.config, 'image_quality', IMAGE_QUALITY)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        return base64.b64encode(bytes(buf)).decode("utf-8")

    # =========================================================================
    # Frame Extraction Methods
    # =========================================================================

    def extract_frames(self, video_path: str) -> List[Tuple[float, str]]:
        """Extract frames from video at specified FPS and convert to base64.

        Uses sequential cap.read() + frame counter instead of cap.set() seeks.
        Seeking in compressed video (H.264/HEVC) forces keyframe decoding on
        every call and is the primary cause of slow extraction on long files.
        """
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
        skip_frames = max(1, int(round(video_fps / self.config.fps)))
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                current_time = frame_count / video_fps
                rotated = self.rotate_frame(frame)
                prepared = self.prepare_frame_for_analysis(rotated)
                base64_frame = self._compress_frame(prepared)
                frames.append((current_time, base64_frame))

                if len(frames) % 50 == 0:
                    logger.info(f"Extracted {len(frames)} frames (timestamp: {current_time:.2f}s)")

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames total")
        return frames

    def extract_frames_phase2(self, video_path: str) -> List[Tuple[float, str]]:
        """Extract frames for Phase 2 verification - always uses PNG format for best quality.

        Uses sequential cap.read() + frame counter (same rationale as extract_frames).
        """
        logger.info(f"Extracting Phase 2 frames from {video_path} at {self.config.fps} FPS (using {self.config.phase2_image_format} format)")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        logger.info(f"Video: {video_fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")

        frames = []
        skip_frames = max(1, int(round(video_fps / self.config.fps)))
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                current_time = frame_count / video_fps
                rotated = self.rotate_frame(frame)
                prepared = self.prepare_frame_for_analysis(rotated)
                base64_frame = self._compress_frame(prepared, format_override=self.config.phase2_image_format)
                frames.append((current_time, base64_frame))

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
        current_batch = FrameBatch(0, self.config.max_batch_size_mb)
        
        for timestamp, base64_frame in frames:
            frame_size = len(base64_frame)
            
            if not current_batch.can_add_frame(frame_size):
                if current_batch.frames:
                    overlap_frames = []
                    if self.config.batch_overlap_frames > 0:
                        overlap_frames = current_batch.frames[-self.config.batch_overlap_frames:]
                    
                    batches.append(current_batch)
                    logger.info(f"Batch {current_batch.batch_id}: {len(current_batch.frames)} frames, {current_batch.get_size_mb():.2f} MB")
                    
                    current_batch = FrameBatch(len(batches), self.config.max_batch_size_mb)
                    
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
        
        content = [
            {"type": "text", "text": PHASE1_SYSTEM_PROMPT.format(context=context)},
        ]

        # Determine MIME type based on configured image format
        mime_type = "image/png" if self.config.image_format == "PNG" else "image/jpeg"

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

            raw_detections = result.get("detections", [])
            logger.info(f"Batch {batch.batch_id}: {len(raw_detections)} raw detections")
            for i, det in enumerate(raw_detections):
                confidence = det.get("confidence", 0) or 0
                filter_status = "PASS" if confidence >= self.config.confidence_threshold else f"FILTERED (conf={confidence})"
                logger.debug(f"  Det {i+1}: conf={confidence}, scale={det.get('scale')}, reading={det.get('scale_reading')}, time={det.get('start_time')}-{det.get('end_time')} [{filter_status}]")
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            batch_token_info = {
                "batch_id": batch.batch_id,
                "frames": len(batch.frames),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": tokens_used
            }
            self.token_usage["phase1_batches"].append(batch_token_info)
            self.token_usage["total_phase1_tokens"] += tokens_used
            self.token_usage["total_tokens"] += tokens_used
            
            logger.info(f"Batch {batch.batch_id} complete. Tokens: {tokens_used}")
            return result
        except Exception as e:
            logger.error(f"Error analyzing batch {batch.batch_id}: {e}")
            return {"detections": [], "context_summary": ""}

    def run_phase1(self, video_path: str) -> List[Detection]:
        """Run phase 1: Initial detection."""
        logger.info(f"PHASE 1: Scanning {video_path}")
        all_frames = self.extract_frames(video_path)
        self.all_frames = all_frames  # keep for Phase 2 frame re-use
        batches = self.create_batches(all_frames)
        
        detections = []
        context = "This is the first batch of the video."
        
        for batch in batches:
            result = self.analyze_batch_phase1(batch, context)
            for det in result.get("detections", []):
                confidence = det.get("confidence", 0) or 0
                if confidence >= self.config.confidence_threshold:
                    # Handle None values explicitly (API may return null for missing fields)
                    raw_start = det.get("start_time")
                    raw_end = det.get("end_time")
                    start_time = float(raw_start) if raw_start is not None else 0.0
                    end_time = float(raw_end) if raw_end is not None else 0.0
                    scale = det.get("scale", "").lower() if det.get("scale") else None

                    # Repair missing/inverted timestamps instead of discarding the detection.
                    # A typical weighing event lasts ~15 s; use that as a fallback duration.
                    _DEFAULT_EVENT_DURATION = 100.0
                    if raw_end is None or end_time <= start_time:
                        if raw_end is None:
                            logger.warning(
                                f"Detection at {start_time:.2f}s has no end_time — "
                                f"inferring end_time as start + {_DEFAULT_EVENT_DURATION:.0f}s"
                            )
                        else:
                            logger.warning(
                                f"Detection has inverted interval {start_time:.2f}s - {end_time:.2f}s — "
                                f"inferring end_time as start + {_DEFAULT_EVENT_DURATION:.0f}s"
                            )
                        end_time = start_time + _DEFAULT_EVENT_DURATION

                    detections.append(Detection(
                        start_time=start_time,
                        end_time=end_time,
                        confidence=float(confidence) if confidence is not None else 0.0,
                        description=det.get("description", "") or "",
                        phase=1,
                        scale=scale,
                        scale_reading=det.get("scale_reading"),
                        unit=det.get("unit"),
                        reading_state=det.get("reading_state")
                    ))
            
            context_summary = result.get("context_summary", "")
            if context_summary:
                self.context_history.append(context_summary)
                context = " | ".join(self.context_history[-1:])
        
        logger.info(f"Phase 1 complete: {len(detections)} potential intervals")
        self.phase1_detections = detections

        # Save representative frames per detection for dashboard display when Phase 2 is off
        for idx, det in enumerate(detections):
            win_start = max(0.0, det.start_time - self.config.clip_buffer_seconds)
            win_end   = det.end_time + self.config.clip_buffer_seconds
            event_frames = [
                (ts, b64) for ts, b64 in all_frames
                if win_start <= ts <= win_end
            ]
            if not event_frames:
                continue
            p1_dir = os.path.join(
                self.config.output_dir, "phase1_event_frames", f"event_{idx + 1}"
            )
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
            # Use specific encoding and error handling to avoid UnicodeDecodeError on Windows
            subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
            logger.info(f"Saved clip: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error clipping video: {e.stderr}")
            raise

    # =========================================================================
    # Phase 2 Analysis
    # =========================================================================

    def _upload_clip_to_gemini(self, clip_path: str):
        """Upload an MP4 clip to the Gemini File API and wait until it is active."""
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized. Check GOOGLE_API_KEY.")
        logger.info(f"Uploading clip to Gemini File API: {os.path.basename(clip_path)}")
        video_file = self.gemini_client.files.upload(
            path=clip_path,
            config={"mime_type": "video/mp4", "display_name": os.path.basename(clip_path)},
        )
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = self.gemini_client.files.get(name=video_file.name)
        if video_file.state.name != "ACTIVE":
            raise ValueError(f"Gemini file upload failed: state={video_file.state.name}")
        logger.info(f"Clip active on Gemini File API: {video_file.name}")
        return video_file

    def analyze_clip_phase2(self, clip_frames: List[Tuple[float, str]], detection_context: str, clip_index: int, phase1_detection: Detection, video_file=None) -> Dict[str, Any]:
        """Analyze a clipped segment in phase 2 using Gemini."""
        logger.info(f"Analyzing clip (Phase 2) with Gemini model: {self.config.phase2_model_name}")
        
        # Include Phase 1 reading and ORIGINAL VIDEO timestamps in context
        phase1_info = (
            f"Phase 1 detected reading: {phase1_detection.scale_reading} {phase1_detection.unit or ''}\n"
            f"Original video time interval: {phase1_detection.start_time:.2f}s to {phase1_detection.end_time:.2f}s\n"
            f"Scale identified: {phase1_detection.scale or 'unknown'}"
        )
        full_context = f"{detection_context}\n{phase1_info}"
        
        system_instruction = PHASE2_SYSTEM_PROMPT.format(context=full_context)
        
        content = []
        content.append(
            f"Verify if this clip contains a valid pork weighing event.\n"
            f"ORIGINAL VIDEO TIME: {phase1_detection.start_time:.2f}s to {phase1_detection.end_time:.2f}s\n"
            f"Phase 1 detected reading: {phase1_detection.scale_reading} {phase1_detection.unit or ''}\n"
            f"Phase 1 detected scale: {phase1_detection.scale or 'unknown'}\n"
        )

        if video_file is not None:
            # Full video clip uploaded via Gemini File API
            content.append(video_file)
        else:
            # Fallback: individual PNG frames
            content.append(f"{len(clip_frames)} frames provided below.")
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
                result_text = result_text.replace("```json", "").replace("```", "")

            result = json.loads(result_text)

            # Track token usage
            tokens_used = 0
            try:
                if hasattr(response, 'usage_metadata'):
                    tokens_used = response.usage_metadata.total_token_count
                    clip_token_info = {
                        "clip_index": clip_index,
                        "frames": len(clip_frames),
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": tokens_used
                    }
                    self.token_usage["phase2_clips"].append(clip_token_info)
                    self.token_usage["total_phase2_tokens"] += tokens_used
                    self.token_usage["total_tokens"] += tokens_used
            except Exception as token_err:
                logger.warning(f"Could not extract token usage from Gemini response: {token_err}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in phase 2 analysis (Gemini): {e}")
            return {"is_valid": False, "confidence": 0.0, "reasoning": f"Gemini Error: {e}"}

    def consolidate_detections(self, detections: List[Detection]) -> List[Detection]:
        """Consolidate overlapping detections, grouped by scale number.

        Scales are numbered 1, 2, … left-to-right. Only merge overlapping
        intervals that belong to the same scale.
        """
        if not detections:
            return []

        # Group detections by scale value dynamically (works for any number of scales)
        scale_groups: dict = {}
        for d in detections:
            key = d.scale if d.scale else "unknown"
            scale_groups.setdefault(key, []).append(d)

        def consolidate_group(group_detections: List[Detection]) -> List[Detection]:
            """Consolidate a single group of detections."""
            if not group_detections:
                return []
            
            sorted_detections = sorted(group_detections, key=lambda d: d.start_time)
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
                    # Keep the best reading
                    if detection.scale_reading and (not current.scale_reading or detection.confidence > current.confidence):
                        current.scale_reading = detection.scale_reading
                        current.unit = detection.unit
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
            return consolidated
        
        # Consolidate each scale group separately
        all_consolidated = []
        log_parts = []
        for scale_key in sorted(scale_groups.keys()):
            group = scale_groups[scale_key]
            consolidated = consolidate_group(group)
            all_consolidated.extend(consolidated)
            log_parts.append(f"Scale {scale_key}: {len(group)}->{len(consolidated)}")

        all_consolidated.sort(key=lambda d: d.start_time)
        logger.info(f"Consolidation by scale: {', '.join(log_parts)}")
        
        return all_consolidated

    def run_phase2(self, video_path: str, phase1_detections: List[Detection]) -> List[Detection]:
        """Run phase 2: Verification using already-extracted Phase 1 frames (no ffmpeg required).

        Overlapping intervals from Phase 1 are merged, then frames that fall
        within each detection window (± clip_buffer_seconds) are filtered from
        self.all_frames and sent directly to Gemini as images.
        """
        logger.info("PHASE 2: Verification")
        consolidated = self.consolidate_detections(phase1_detections)

        logger.info(f"Consolidated {len(phase1_detections)} Phase 1 detections into {len(consolidated)} events for verification")

        # Fall back to re-extracting if Phase 1 frames were never stored
        source_frames: List[Tuple[float, str]] = getattr(self, "all_frames", None) or []
        if not source_frames:
            logger.warning("Phase 1 frames not cached; re-extracting for Phase 2")
            source_frames = self.extract_frames_phase2(video_path)

        verified = []

        for idx, detection in enumerate(consolidated):
            logger.info(f"Verifying {idx+1}/{len(consolidated)}: {detection.start_time:.1f}s-{detection.end_time:.1f}s")

            # Filter frames that fall inside the detection window (with buffer)
            buf = self.config.clip_buffer_seconds
            win_start = max(0.0, detection.start_time - buf)
            win_end = detection.end_time + buf
            clip_frames = [
                (ts - detection.start_time, b64)   # convert to clip-relative timestamps
                for ts, b64 in source_frames
                if win_start <= ts <= win_end
            ]
            logger.info(f"  Using {len(clip_frames)} frames from [{win_start:.1f}s, {win_end:.1f}s] window")

            # Save up to 5 evenly-spaced verification frames for dashboard display
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

            context = f"Potential pork weighing event: {detection.description}"
            result = self.analyze_clip_phase2(clip_frames, context, idx + 1, detection)

            # Save response
            with open(os.path.join(self.responses_dir, f"response_{idx+1}.json"), 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            if result.get("is_valid", False):
                # Ensure float conversion for readings
                def to_float(val, default):
                    try:
                        return float(val) if val is not None else default
                    except (ValueError, TypeError):
                        return default

                # Use Phase 1 timestamps directly (Phase 2 only verifies, doesn't refine timestamps)
                verified_det = Detection(
                    start_time=detection.start_time,
                    end_time=detection.end_time,
                    confidence=float(result.get("confidence", 0)),
                    description=result.get("description", ""),
                    phase=2,
                    scale=result.get("scale", "").lower() if result.get("scale") else detection.scale,
                    scale_reading=to_float(result.get("verified_reading", None), detection.scale_reading),
                    unit=result.get("unit") or detection.unit,
                    reading_state=result.get("reading_state"),
                    is_valid=True,
                    reasoning=result.get("reasoning", ""),
                    verified_reading=to_float(result.get("verified_reading", None), None),
                    reading_correction=result.get("reading_correction")
                )
                verified.append(verified_det)
                logger.info(f"[OK] VERIFIED: Scale={verified_det.scale}, Reading = {result.get('verified_reading')} {result.get('unit', '')}, Time={detection.start_time:.1f}s-{detection.end_time:.1f}s")
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

    def create_results_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from detection results."""
        detections = self.phase2_detections if self.config.enable_phase2 else self.phase1_detections
        
        data = {
            "event_id": [],
            "start_time": [],
            "end_time": [],
            "scale": [],
            "phase1_reading": [],
            "verified_reading": [],
            "unit": [],
            "reading_state": [],
            "confidence": [],
            "description": [],
            "reading_correction": []
        }
        
        for idx, det in enumerate(detections, 1):
            data["event_id"].append(idx)
            data["start_time"].append(det.start_time)
            data["end_time"].append(det.end_time)
            data["scale"].append(det.scale)
            data["phase1_reading"].append(det.scale_reading)
            data["verified_reading"].append(det.verified_reading if hasattr(det, 'verified_reading') else det.scale_reading)
            data["unit"].append(det.unit)
            data["reading_state"].append(det.reading_state)
            data["confidence"].append(det.confidence)
            data["description"].append(det.description)
            data["reading_correction"].append(det.reading_correction if hasattr(det, 'reading_correction') else None)
        
        return pd.DataFrame(data)

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
        
        # Save CSV
        df = self.create_results_dataframe()
        csv_path = os.path.join(self.config.output_dir, "weighing_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV results saved to: {csv_path}")
        
        # Text summary
        summary_path = os.path.join(self.config.output_dir, "analysis_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"PORK WEIGHING ANALYSIS SUMMARY\n{'='*50}\n")
            f.write(f"Video: {self.config.input_video_path}\n")
            f.write(f"Analysis FPS: {self.config.fps}\n")
            f.write(f"Video FPS: {self.video_fps:.2f}\n")
            f.write(f"ROI: {self.roi}\n")
            f.write(f"Rotation: {self.config.rotation_angle} degrees\n")
            f.write(f"Cropping: {'Enabled' if self.config.enable_cropping else 'Disabled (Box overlay)'}\n")
            f.write(f"Upscale Factor: {self.config.image_upscale_factor}x\n\n")
            f.write(f"Phase 1 Model: {self.config.phase1_model_name}\n")
            if self.config.enable_phase2:
                f.write(f"Phase 2 Model: {self.config.phase2_model_name}\n")
            f.write(f"\nPhase 1 Detections: {len(self.phase1_detections)}\n")
            f.write(f"Phase 2 Verified: {len(self.phase2_detections)}\n\n")
            f.write(f"Total Tokens: {self.token_usage['total_tokens']}\n")
            f.write(f"  - Phase 1: {self.token_usage['total_phase1_tokens']}\n")
            f.write(f"  - Phase 2: {self.token_usage['total_phase2_tokens']}\n\n")
            f.write("-" * 50 + "\n")
            f.write("VERIFIED WEIGHING EVENTS:\n")
            f.write("-" * 50 + "\n")
            
            detections = self.phase2_detections if self.config.enable_phase2 else self.phase1_detections
            for idx, d in enumerate(detections, 1):
                f.write(f"\n{idx}. Time: {d.start_time:.1f}s - {d.end_time:.1f}s\n")
                reading = d.verified_reading if hasattr(d, 'verified_reading') and d.verified_reading else d.scale_reading
                f.write(f"   Reading: {reading} {d.unit or ''}\n")
                f.write(f"   Confidence: {d.confidence:.2f}\n")
                f.write(f"   State: {d.reading_state}\n")
                f.write(f"   Description: {d.description}\n")
                if hasattr(d, 'reading_correction') and d.reading_correction:
                    f.write(f"   Correction: {d.reading_correction}\n")
        
        logger.info(f"Summary saved to: {summary_path}")

    # =========================================================================
    # Main Run Method
    # =========================================================================

    def run(self):
        """Main pipeline execution."""
        try:
            # Step 1: ROI Selection
            print("\n" + "="*60)
            print("PORK WEIGHING ANALYSIS PIPELINE")
            print("="*60)
            print(f"Video: {self.config.input_video_path}")
            print(f"Phase 1 Model: {self.config.phase1_model_name}")
            if self.config.enable_phase2:
                print(f"Phase 2 Model: {self.config.phase2_model_name}")
            print("="*60)
            
            if not self.roi:
                print("\nStep 1: Select the Region of Interest (ROI)")
                print("This should be the area containing the weighing scale display.")
                if not self.select_roi():
                    logger.error("ROI selection cancelled. Exiting.")
                    return
            
            # Step 2: Phase 1 Analysis
            print("\nStep 2: Running Phase 1 Analysis (Event Detection + Initial OCR)...")
            self.run_phase1(self.config.input_video_path)
            
            # Step 3: Phase 2 Verification (if enabled)
            if self.config.enable_phase2:
                logger.info("Two-phase analysis mode: Running Phase 2 verification...")
                if self.phase1_detections:
                    print("\nStep 3: Running Phase 2 Verification...")
                    self.run_phase2(self.config.input_video_path, self.phase1_detections)
                    self.create_merged_video(self.config.input_video_path)
                else:
                    logger.info("No Phase 1 detections found. Skipping Phase 2.")
            else:
                logger.info("Single-phase analysis mode: Skipping Phase 2 verification.")
                self.phase2_detections = self.phase1_detections
            
            # Step 4: Save Results
            print("\nStep 4: Saving results...")
            self.save_results()
            
            # Print summary
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
    parser = argparse.ArgumentParser(description="Pork Weighing Analysis Pipeline")
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--fps", type=float, default=FPS, help="Frames per second to analyze")
    parser.add_argument("--no-phase2", action="store_true", help="Disable Phase 2 verification")
    args = parser.parse_args()
    
    # Determine video path
    video_path = args.video_path or DEFAULT_INPUT_VIDEO
    
    if not video_path:
        print("Error: No video path provided.")
        print("Usage: python pork_weighing_analysis.py <video_path>")
        print("OR set INPUT_VIDEO_PATH in .env file")
        return
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Check API keys
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set in .env file!")
        return
    
    if not GOOGLE_API_KEY and not args.no_phase2 and ENABLE_PHASE2:
        print("\n[WARNING] GOOGLE_API_KEY is not set in .env!")
        print("Phase 2 analysis will fail. Add --no-phase2 to run single-phase analysis.")
        print("-" * 60)
    
    # Create unique output directory
    video_name = Path(video_path).stem
    output_dir = get_unique_output_dir(args.output, video_name)
    
    # Create config
    config = PipelineConfig(
        input_video_path=video_path,
        output_dir=output_dir,
        fps=args.fps,
        enable_phase2=not args.no_phase2 and ENABLE_PHASE2
    )
    
    # Run pipeline
    pipeline = PorkWeighingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
