"""
Customer Service Time Analysis Pipeline with Two-Phase Detection
---------------------------------------------------------
A unified pipeline to track individual customers and calculate service time.

Phase 1: GPT-4o-mini - Detect customer entry and ramen serving events
Phase 2: Gemini-2.0-flash - Verify events and confirm service timing
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
    import pandas as pd
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install requirements: pip install openai opencv-python-headless numpy pillow google-generativeai python-dotenv pandas")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# ⚙️ AGENT-LEVEL PARAMETER DEFAULTS
# Canonical defaults for the avg serve time task.
# webapp/config.py can import these when this agent is active.
# ============================================================================

AGENT_PHASE1_MODEL_NAME       = "gpt-5-mini"
AGENT_PHASE2_MODEL_NAME       = "gemini-2.5-pro"
AGENT_FPS                     = 1.0
AGENT_CONFIDENCE_THRESHOLD    = 0.7
AGENT_MAX_BATCH_SIZE_MB       = 35.0
AGENT_CLIP_BUFFER_SECONDS     = 3
AGENT_MAX_FRAMES_PER_BATCH    = 300
AGENT_BATCH_OVERLAP_FRAMES    = 5
AGENT_IMAGE_QUALITY           = 95
AGENT_IMAGE_UPSCALE_FACTOR    = 1.0
AGENT_IMAGE_TARGET_RESOLUTION = "auto"
AGENT_IMAGE_FORMAT            = "JPEG"   # JPEG acceptable for serve-time (no OCR)
AGENT_PHASE2_IMAGE_FORMAT     = "PNG"
AGENT_IMAGE_INTERPOLATION     = "LANCZOS"
AGENT_ENABLE_CROPPING         = True
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
IMAGE_FORMAT            = AGENT_IMAGE_FORMAT
PHASE2_IMAGE_FORMAT     = AGENT_PHASE2_IMAGE_FORMAT
IMAGE_INTERPOLATION     = os.getenv("IMAGE_INTERPOLATION",        AGENT_IMAGE_INTERPOLATION).upper()
PHASE1_MAX_LONG_EDGE: int = int(os.getenv("PHASE1_MAX_LONG_EDGE", "448"))

# Cropping / Rotation
ENABLE_CROPPING = os.getenv("ENABLE_CROPPING", str(AGENT_ENABLE_CROPPING)).lower() == "true"
ROTATION_ANGLE  = int(os.getenv("ROTATION_ANGLE", str(AGENT_ROTATION_ANGLE)))

# Input/Output Defaults
DEFAULT_INPUT_VIDEO = os.getenv("INPUT_VIDEO_PATH", "")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./results_service_time")

# ============================================================================
# 🤖 PROMPTS
# ============================================================================

PHASE1_SYSTEM_PROMPT = """You are a video analysis assistant specialized in measuring customer service time in a ramen restaurant. Your task is to identify WHEN A CUSTOMER BECOMES SEATED AND STATIONARY at a counter seat (START), and WHEN THE CHEF PLACES A COMPLETED RAMEN BOWL ON THE SILVER GREY COUNTER FOR THAT CUSTOMER (END), in order to calculate average service time per customer.

CRITICAL - TIMESTAMP READING:
- Read timestamps from the BOTTOM-RIGHT CORNER of each frame
- Use these exact timestamps for customer seated time and bowl placement time
- Format timestamps as they appear in the frames

WHAT IS THE SILVER GREY COUNTER:
- The silver/shiny grey stainless-steel pass-through surface between the kitchen and the customer seating area
- The chef places completed ramen bowls here for customers to receive their order
- It is NOT the cooking/prep surface — it is specifically the customer-facing service counter

WHAT TO DETECT — Service Time Events:
1. **CUSTOMER SEATED** (START): A new customer sits down and becomes stationary at a fixed counter seat
   - Customer must be clearly seated and stationary — not just passing by or standing temporarily
   - Record exact timestamp from bottom-right corner when customer is settled at their seat
   - Note the customer's clothing/appearance and seat position
   - **EXCLUDE** customers who were already seated at the very start of the video (no clear seating event observed)

2. **BOWL PLACED ON SILVER GREY COUNTER** (END): The chef places a completed ramen bowl on the silver grey counter in front of / for this customer
   - Chef physically sets down a light blue ramen bowl on the silver grey counter
   - This must be on the silver/shiny grey counter specifically — NOT on a prep surface, cooking area, or handed directly to the customer
   - Record exact timestamp from bottom-right corner at this moment

3. **SERVICE TIME**: The elapsed time from customer seated (START) to bowl placed on silver grey counter (END)
   - This measures how long the customer waits from sitting down until their food is ready
   - Sanity check: reject if service time < 1 minute (60 seconds) or > 60 minutes (likely a mis-detection)

CUSTOMER TRACKING:
- Assign each customer a UNIQUE ID based on clothing/appearance (e.g., "customer_1_red_shirt")
- Track each newly-seated customer until their bowl is placed on the counter

ANTI-HALLUCINATION RULES:
- Only report a detection when BOTH the customer seating event AND the bowl placement on the silver grey counter are clearly observed in these frames
- Do NOT detect customers who were already seated at the start of the video
- Do NOT detect bowl placements on surfaces other than the silver grey counter
- If a customer is seated but their bowl has not yet been placed, track them in active_customers for the next batch

Context from previous batch: {context}

IMPORTANT: Review context_summary and active_customers. Continue tracking seated customers whose bowl has not yet been placed on the counter.

For each complete service event (customer seated AND bowl placed on silver grey counter), provide:
- customer_id: Unique identifier (e.g., "customer_1_red_shirt")
- bowl_arrival_time: Exact timestamp from bottom-right when customer became seated/stationary
- return_time: Exact timestamp from bottom-right when chef placed bowl on silver grey counter
- bowl_arrival_video_time: Video time in seconds when customer sat down
- return_video_time: Video time in seconds when bowl was placed on silver grey counter
- bowl_completion_status: Always "N/A" for this task
- remaining_contents: Brief description of the event (e.g., "Customer in red shirt seated at position 3, chef placed blue bowl on silver counter after 8 mins")
- eating_duration_seconds: Service time in seconds (bowl placement time minus customer seated time)
- seat_position: Counter seat position where the customer is sitting
- appearance_description: Detailed customer appearance (clothing, accessories)
- confidence: 0.0 to 1.0
- description: Brief description of the complete service time event

Return your response in JSON format:
{{
  "detections": [
    {{
      "customer_id": "<unique identifier with appearance>",
      "bowl_arrival_time": "<timestamp from bottom-right when customer seated>",
      "return_time": "<timestamp from bottom-right when bowl placed on silver grey counter>",
      "bowl_arrival_video_time": <video seconds when customer sat down>,
      "return_video_time": <video seconds when bowl placed on counter>,
      "bowl_completion_status": "N/A",
      "remaining_contents": "<brief description of the service event>",
      "eating_duration_seconds": <service time in seconds>,
      "seat_position": "<counter seat position>",
      "appearance_description": "<detailed clothing and appearance of customer>",
      "confidence": <0.0-1.0>,
      "description": "<brief description of the complete service time event>"
    }}
  ],
  "active_customers": [
    {{
      "customer_id": "<unique identifier>",
      "bowl_arrival_time": "<timestamp when customer seated>",
      "bowl_arrival_video_time": <seconds>,
      "seat_position": "<location>",
      "appearance_description": "<clothing and appearance>",
      "last_observed_timestamp": "<latest timestamp where customer was seen waiting>",
      "description": "<brief status — customer seated, waiting for bowl>"
    }}
  ],
  "context_summary": "<summary of: 1) customers currently seated and waiting for their bowl, 2) any service events just completed, 3) timestamp range of this batch>"
}}
"""

PHASE2_SYSTEM_PROMPT = """You are a video analysis assistant verifying customer service time events in a ramen restaurant. You are analyzing frames from two windows: one around the customer seating event and one around the bowl placement event.

Your task: verify that (1) the customer truly became seated and stationary at a counter seat, and (2) the chef truly placed a completed ramen bowl on the silver grey counter for that customer, and confirm the service time.

CRITICAL - TIMESTAMP VERIFICATION:
- Read and verify timestamps from the BOTTOM-RIGHT CORNER of frames
- Confirm CUSTOMER SEATED TIME: when the customer was settled and stationary at their seat
- Confirm BOWL PLACEMENT TIME: when the chef placed the ramen bowl on the silver grey counter
- SERVICE TIME = BOWL PLACEMENT TIME - CUSTOMER SEATED TIME (in seconds)

THE SILVER GREY COUNTER:
- The silver/shiny grey stainless-steel pass-through counter where the chef places completed orders
- NOT the cooking/prep surface — this is specifically the customer-facing service counter

VERIFICATION CRITERIA:

1. **CUSTOMER SEATED**: Was the customer truly newly seated and stationary?
   - Customer is clearly seated and settled at a fixed counter position
   - REJECT if the customer was already seated at the very start of the clip (no clear new-seating event)
   - REJECT if the customer is just standing or passing by, not seated

2. **BOWL PLACED ON SILVER GREY COUNTER**: Did the chef truly place a bowl on the silver grey counter?
   - Chef physically sets down a ramen bowl (typically light blue) on the silver/shiny grey counter
   - The surface must be the silver grey service counter — REJECT if placed elsewhere (prep surface, handed directly, etc.)
   - REJECT if the bowl was already on the counter at the start of the clip

3. **CUSTOMER IDENTITY**: Verify the same customer from Phase 1 detection
   - Check clothing/appearance matches the described customer
   - The bowl placement must correspond to this specific seated customer

CRITICAL REJECTION CRITERIA — AUTO-REJECT if:
- Customer seating event not clearly observed
- Bowl was NOT placed on the silver grey counter (placed elsewhere or handed directly)
- Service time < 60 seconds or > 3600 seconds / 60 minutes (sanity bounds)
- Cannot confirm it is the silver grey counter surface

Previous detection context: {context}

Analyze the provided frames carefully. Focus on: (1) confirming the customer was newly seated and stationary, and (2) confirming the chef placed the bowl on the silver grey counter.

Return your response in JSON format:
{{
  "is_valid": <true/false>,
  "customer_id_verified": "<confirmed customer_id or appearance description>",
  "appearance_match_confirmed": <true/false>,
  "verified_bowl_arrival_time": "<timestamp from bottom-right when customer was seated, or null>",
  "verified_return_time": "<timestamp from bottom-right when bowl placed on silver grey counter, or null>",
  "eating_duration_seconds": <service time in seconds, or null>,
  "bowl_completion_status": "N/A",
  "remaining_contents": "<brief description — e.g., customer in red shirt seated at pos 3, chef placed blue bowl on silver counter after 8 min>",
  "return_event_confirmed": <true/false — true if bowl placement on silver grey counter confirmed>,
  "customer_tracking_consistent": <true/false>,
  "timestamp_quality": "<clear/partial/unclear>",
  "confidence": <0.0-1.0>,
  "appearance_details_verified": "<description of customer appearance>",
  "tracking_consistency_notes": "<notes on customer identity verification and silver grey counter confirmation>",
  "description": "<detailed description: customer seated at time X, chef placed bowl on silver grey counter at time Y, service time Z seconds>",
  "reasoning": "<why valid/invalid, confirmation of new seating event, confirmation of silver grey counter placement, service time calculation, confidence justification>"
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
    _fh = logging.FileHandler('service_time_analysis.log', encoding='utf-8', mode='w')
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
    clip_buffer_seconds: int = CLIP_BUFFER_SECONDS
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    batch_overlap_frames: int = BATCH_OVERLAP_FRAMES
    phase1_model_name: str = PHASE1_MODEL_NAME
    phase2_model_name: str = PHASE2_MODEL_NAME
    enable_phase2: bool = ENABLE_PHASE2
    enable_cropping: bool = ENABLE_CROPPING
    rotation_angle: int = ROTATION_ANGLE
    image_upscale_factor: float = IMAGE_UPSCALE_FACTOR
    image_target_resolution: str = IMAGE_TARGET_RESOLUTION  # "auto" or "width,height"
    image_format: str = IMAGE_FORMAT  # "PNG" or "JPEG"
    phase2_image_format: str = PHASE2_IMAGE_FORMAT  # Phase 2 format (default PNG)
    image_interpolation: str = IMAGE_INTERPOLATION  # "LANCZOS" or "CUBIC"
    phase1_max_long_edge: int = PHASE1_MAX_LONG_EDGE
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
    """Represents a detected customer service event."""
    customer_id: str
    seated_time: str  # OCR timestamp from bottom-right
    serving_time: str # OCR timestamp from bottom-right 
    video_seated_time: float # seconds from video start for clipping
    video_serving_time: float # seconds from video start for clipping
    confidence: float
    description: str
    phase: int  # 1 or 2
    appearance_description: Optional[str] = None
    tracking_notes: Optional[str] = None
    is_valid: Optional[bool] = None  # Set in phase 2
    appearance_match_confirmed: Optional[bool] = None
    service_time_seconds: Optional[float] = None
    reasoning: Optional[str] = None
    seated_verified: Optional[bool] = None
    serving_verified: Optional[bool] = None
    bowl_color_confirmed: Optional[str] = None
    customer_tracking_consistent: Optional[bool] = None
    timestamp_quality: Optional[str] = None
    bowl_completion_status: str = "UNKNOWN"  # COMPLETED / NOT_COMPLETED / UNCERTAIN
    remaining_contents: Optional[str] = None


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
        
    def mouse_callback(self, event, x, y, flags, param):
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
        print("1. Draw a rectangle around the customer entrance and seating area")
        print("2. Click and drag to draw the rectangle")
        print("3. Press 'c' or ENTER to CONFIRM the selection")
        print("4. Press 'r' to RESET and redraw")
        print("5. Press 'q' or ESC to QUIT without selection")
        print("="*60 + "\n")
        
        while True:
            display_frame = self.frame.copy()
            cv2.putText(display_frame, "Draw rectangle around entrance/seating area | C: Confirm | R: Reset | Q: Quit", 
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

class CustomerServicePipeline:
    """Main pipeline for customer service time tracking."""
    
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
        self.active_customers: List[Dict[str, Any]] = []
        self.phase1_detections: List[Detection] = []
        self.phase2_detections: List[Detection] = []
        self.temp_dir = tempfile.mkdtemp(prefix="service_time_")
        
        # ROI storage
        self.roi = config.roi
        self.video_fps = None

        # Pre-compute ffmpeg path (avoid repeated searches per Phase 2 clip)
        try:
            self._ffmpeg_path = self._find_ffmpeg()
        except FileNotFoundError:
            self._ffmpeg_path = None

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
        self.event_frames_dir = os.path.join(config.output_dir, "event_frames")
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.event_frames_dir, exist_ok=True)
        
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

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            # Use shutil.rmtree with error handling to avoid issues with open files on Windows
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp dir: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir {self.temp_dir}: {e}")

    @staticmethod
    def _robust_float(val: Any, default: float = 0.0) -> float:
        """Robustly convert a value to float, handling 's' suffix and other noise."""
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        try:
            # String handling: remove 's' suffix, whitespace, and handle common issues
            clean_str = str(val).lower().strip().replace('s', '')
            if not clean_str or clean_str in ['n/a', 'none', 'null', '-']:
                return default
            return float(clean_str)
        except (ValueError, TypeError):
            return default

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
        selector = ROISelector(frame, "Select Customer Entrance/Seating Region")
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

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame so the ROI occupies 1/3 of the resulting image in each dimension.

        The crop is centred on the bounding box. At the 1/3 rule the crop
        window is 3× the bbox width and 3× the bbox height, so the bbox
        covers exactly 1/3 of the crop width and 1/3 of the crop height,
        giving meaningful surrounding context without blurring small bboxes.
        """
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

    _MAX_UPSCALED_LONG_EDGE = 1920  # cap to avoid OOM on large ROIs

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
        """Prepare frame for OCR/Analysis based on enable_cropping setting."""
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

    def capture_event_frames(self, video_path: str, detections: List[Detection], phase_label: str = "phase1"):
        """Capture and save the exact frames at detected start and end times for each event.
        
        For each detection, extracts the frame closest to:
          - video_seated_time (customer seated / start time)
          - video_serving_time (ramen served / end time)
        Saves them as PNG images in the event_frames directory with annotations.
        
        Args:
            video_path: Path to the original video file.
            detections: List of Detection objects with video timestamps.
            phase_label: Label prefix for saved files (e.g., 'phase1' or 'phase2').
        """
        if not detections:
            logger.info(f"No detections to capture event frames for ({phase_label}).")
            return
        
        logger.info(f"Capturing event frames for {len(detections)} detections ({phase_label})...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video for event frame capture: {video_path}")
            return
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps <= 0:
            logger.error(f"Invalid video FPS ({video_fps}) for event frame capture.")
            cap.release()
            return
        
        for idx, detection in enumerate(detections, 1):
            # Collect the target timestamps for this detection
            targets = [
                ("start", detection.video_seated_time, detection.seated_time),
                ("end", detection.video_serving_time, detection.serving_time),
            ]
            
            for event_type, video_time, ocr_time in targets:
                # Calculate the exact frame number for this timestamp
                target_frame_num = int(video_time * video_fps)
                target_frame_num = max(0, min(target_frame_num, total_frames - 1))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Could not read frame at {video_time:.2f}s (frame {target_frame_num}) for {detection.customer_id} {event_type}")
                    continue
                
                # Apply rotation to match analysis view
                frame = self.rotate_frame(frame)
                
                # Create annotated version
                annotated = frame.copy()
                h, w = annotated.shape[:2]
                
                # Draw a semi-transparent banner at the top
                banner_height = 80
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
                
                # Add annotation text
                event_label = "SEATED (Start)" if event_type == "start" else "SERVED (End)"
                color = (0, 255, 0) if event_type == "start" else (0, 165, 255)  # Green for start, Orange for end
                
                text_line1 = f"{event_label} | Customer: {detection.customer_id}"
                text_line2 = f"Video Time: {video_time:.2f}s | OCR Time: {ocr_time} | Conf: {detection.confidence:.2f}"
                
                cv2.putText(annotated, text_line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(annotated, text_line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                
                # Draw ROI rectangle if available
                if self.roi:
                    x1, y1, x2, y2 = self.roi
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Save annotated frame
                safe_customer_id = detection.customer_id.replace(' ', '_').replace('/', '_')[:50]
                filename = f"{phase_label}_event{idx}_{event_type}_{video_time:.1f}s_{safe_customer_id}.png"
                save_path = os.path.join(self.event_frames_dir, filename)
                cv2.imwrite(save_path, annotated)
                
                # Also save the raw (un-annotated) frame
                raw_filename = f"{phase_label}_event{idx}_{event_type}_{video_time:.1f}s_{safe_customer_id}_raw.png"
                raw_save_path = os.path.join(self.event_frames_dir, raw_filename)
                cv2.imwrite(raw_save_path, frame)
                
                logger.info(f"Saved {event_type} frame for {detection.customer_id}: {save_path}")
        
        cap.release()
        logger.info(f"Event frame capture complete ({phase_label}): saved to {self.event_frames_dir}")

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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / video_fps
            
            if current_time >= next_extract_time:
                # Apply rotation
                rotated = self.rotate_frame(frame)
                
                # Prepare frame (crop + upscale or draw box)
                prepared = self.prepare_frame_for_analysis(rotated)
                
                # Convert to base64
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
        """Extract frames for Phase 2 verification - always uses PNG format for best quality."""
        logger.info(f"Extracting Phase 2 frames from {video_path} at {self.config.fps} FPS (using {self.config.phase2_image_format} format)")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        logger.info(f"Video: {video_fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
        
        frames = []
        time_interval = 1.0 / self.config.fps
        next_extract_time = 0.0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / video_fps
            
            if current_time >= next_extract_time:
                # Apply rotation
                rotated = self.rotate_frame(frame)
                
                # Prepare frame (crop + upscale or draw box)
                prepared = self.prepare_frame_for_analysis(rotated)
                
                # Convert to base64 using Phase 2 format (always PNG for best quality)
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
            
            # DEBUG: Log raw API response content
            raw_content = response.choices[0].message.content
            logger.info(f"=" * 60)
            logger.info(f"[DEBUG] BATCH {batch.batch_id} - RAW API RESPONSE:")
            logger.info(f"{raw_content}")
            logger.info(f"=" * 60)
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
            
            # DEBUG: Log parsed detections before filtering
            raw_detections = result.get("detections", [])
            logger.info(f"[DEBUG] Batch {batch.batch_id}: Found {len(raw_detections)} raw detections before filtering")
            print(f"[DEBUG] Batch {batch.batch_id}: Found {len(raw_detections)} raw detections before filtering")
            
            for i, det in enumerate(raw_detections):
                confidence = det.get("confidence", 0) or 0
                customer_id = det.get("customer_id", "N/A")
                bowl_arrival_time = det.get("bowl_arrival_time", "N/A")
                return_time = det.get("return_time", "N/A")
                bowl_status = det.get("bowl_completion_status", "N/A")

                filter_status = "PASSED" if confidence >= self.config.confidence_threshold else f"FILTERED (conf {confidence} < threshold {self.config.confidence_threshold})"

                logger.info(f"  Detection {i+1}: customer={customer_id}, conf={confidence}, arrival={bowl_arrival_time}, return={return_time}, completion={bowl_status}, [{filter_status}]")
                print(f"  Detection {i+1}: customer={customer_id}, conf={confidence}, arrival={bowl_arrival_time}, return={return_time}, completion={bowl_status}, [{filter_status}]")
            
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
        """Run phase 1: Initial detection of customer service events."""
        logger.info(f"PHASE 1: Scanning {video_path}")

        all_frames = self.extract_frames(video_path)
        batches = self.create_batches(all_frames)
        
        detections = []
        context = "This is the first batch of the video."
        self.active_customers = []
        
        for batch in batches:
            # Construct context from active customers and history
            full_context = context
            if self.active_customers:
                 full_context += f" | ACTIVE CUSTOMERS: {json.dumps(self.active_customers)}"
            
            result = self.analyze_batch_phase1(batch, full_context)
            
            # Map batch video timestamps
            batch_start_vid_time = batch.frames[0][0]
            batch_end_vid_time = batch.frames[-1][0]
            
            for det in result.get("detections", []):
                confidence = self._robust_float(det.get("confidence"), 0.0)
                if confidence >= self.config.confidence_threshold:
                    detections.append(Detection(
                        customer_id=det.get("customer_id", "unknown"),
                        seated_time=det.get("bowl_arrival_time", "N/A"),      # bowl arrival → seated_time
                        serving_time=det.get("return_time", "N/A"),           # return → serving_time
                        video_seated_time=self._robust_float(det.get("bowl_arrival_video_time"), batch_start_vid_time),
                        video_serving_time=self._robust_float(det.get("return_video_time"), batch_end_vid_time),
                        confidence=float(confidence),
                        description=det.get("description", ""),
                        phase=1,
                        appearance_description=det.get("appearance_description"),
                        tracking_notes=det.get("remaining_contents"),
                        service_time_seconds=self._robust_float(det.get("eating_duration_seconds")),
                        seated_verified=None,
                        serving_verified=None,
                        bowl_color_confirmed=None,
                        customer_tracking_consistent=None,
                        timestamp_quality=None,
                        bowl_completion_status=det.get("bowl_completion_status", "UNKNOWN"),
                        remaining_contents=det.get("remaining_contents"),
                    ))
            
            # Update tracking status
            self.active_customers = result.get("active_customers", [])
            context_summary = result.get("context_summary", "")
            if context_summary:
                self.context_history.append(context_summary)
                context = context_summary
        
        logger.info(f"Phase 1 complete: {len(detections)} potential service events")
        self.phase1_detections = detections
        
        # Capture and save the exact frames at detected start/end times
        self.capture_event_frames(video_path, detections, phase_label="phase1")
        
        return detections

    # =========================================================================
    # Video Clipping
    # =========================================================================

    def _find_ffmpeg(self) -> str:
        """Find ffmpeg executable path."""
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            common_paths = [r'C:\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe']
            for path in common_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found. Please install FFmpeg to use this pipeline.")
        return ffmpeg_path

    def _validate_clip(self, clip_path: str) -> bool:
        """Check if a clip file exists, has non-zero size, and can be opened by OpenCV."""
        if not os.path.exists(clip_path):
            logger.warning(f"Clip file does not exist: {clip_path}")
            return False
        if os.path.getsize(clip_path) == 0:
            logger.warning(f"Clip file is empty (0 bytes): {clip_path}")
            return False
        cap = cv2.VideoCapture(clip_path)
        opened = cap.isOpened()
        if opened:
            ret, _ = cap.read()
            cap.release()
            if not ret:
                logger.warning(f"Clip file exists but cannot read any frames: {clip_path}")
                return False
        else:
            cap.release()
            logger.warning(f"Clip file exists but OpenCV cannot open it: {clip_path}")
            return False
        return True

    def clip_video_segment(self, video_path: str, start_time: float, end_time: float, output_path: str):
        """Clip a segment from video using ffmpeg.
        
        First attempts fast stream copy (-c copy). If the resulting clip is
        unreadable (e.g. cut at a non-keyframe), falls back to re-encoding.
        """
        start_with_buffer = max(0, start_time - self.config.clip_buffer_seconds)
        end_with_buffer = end_time + self.config.clip_buffer_seconds
        duration = end_with_buffer - start_with_buffer
        
        ffmpeg_path = self._ffmpeg_path
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found. Please install FFmpeg to use this pipeline.")
        
        # Attempt 1: fast stream copy
        cmd_copy = [
            ffmpeg_path, '-i', video_path, '-ss', str(start_with_buffer), '-t', str(duration),
            '-c', 'copy', '-y', output_path
        ]
        
        try:
            subprocess.run(cmd_copy, check=True, capture_output=True, encoding='utf-8', errors='replace')
            logger.info(f"Saved clip (stream copy): {output_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Stream copy clipping failed: {e.stderr}")
        
        # Validate the stream-copied clip
        if self._validate_clip(output_path):
            return
        
        # Attempt 2: re-encode (slower but handles non-keyframe cuts)
        logger.info(f"Stream-copied clip is unreadable, retrying with re-encoding: {output_path}")
        # Place -ss before -i for input seeking (fast seek to nearest keyframe, then decode)
        cmd_reencode = [
            ffmpeg_path, '-ss', str(start_with_buffer), '-i', video_path, '-t', str(duration),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'aac', '-y', output_path
        ]
        
        try:
            subprocess.run(cmd_reencode, check=True, capture_output=True, encoding='utf-8', errors='replace')
            logger.info(f"Saved clip (re-encoded): {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error clipping video (re-encode): {e.stderr}")
            raise
        
        if not self._validate_clip(output_path):
            raise ValueError(f"Failed to create a valid clip even after re-encoding: {output_path}")

    # =========================================================================
    # Phase 2 Analysis
    # =========================================================================

    def analyze_clip_phase2(self, clip_frames: List[Tuple[float, str]], detection_context: str, clip_index: int, phase1_detection: Detection, arrival_frame_count: int = 0) -> Dict[str, Any]:
        """Analyze a clipped segment in phase 2 using Gemini.

        clip_frames may contain frames from two windows (arrival + return).
        arrival_frame_count indicates how many frames belong to the arrival window;
        a separator label is inserted between the two windows.
        """
        logger.info(f"Analyzing clip (Phase 2) with Gemini model: {self.config.phase2_model_name}")

        # Include Phase 1 info in context
        phase1_info = (
            f"Phase 1 detected customer: {phase1_detection.customer_id}\n"
            f"Appearance: {phase1_detection.appearance_description}\n"
            f"Bowl Arrival Time: {phase1_detection.seated_time}\n"
            f"Return Time: {phase1_detection.serving_time}\n"
            f"Phase 1 Completion Status: {phase1_detection.bowl_completion_status}\n"
            f"Phase 1 Remaining Contents: {phase1_detection.remaining_contents or 'N/A'}"
        )
        full_context = f"{detection_context}\n{phase1_info}"

        system_instruction = PHASE2_SYSTEM_PROMPT.format(context=full_context)

        content = []
        window_desc = (
            "Two short windows are provided: BOWL ARRIVAL (±5s) and BOWL RETURN (±5s). "
            "The middle eating period is intentionally omitted."
            if arrival_frame_count > 0 else f"{len(clip_frames)} frames provided below."
        )
        content.append(
            f"Verify if this clip contains a valid bowl return event.\n"
            f"OCR TIMESTAMPS: arrival={phase1_detection.seated_time}, return={phase1_detection.serving_time}\n"
            f"Phase 1 completion assessment: {phase1_detection.bowl_completion_status}\n"
            f"Phase 1 detected customer: {phase1_detection.customer_id}\n"
            f"{window_desc}"
        )

        for i, (timestamp, base64_frame) in enumerate(clip_frames):
            # Insert separator between the two windows
            if arrival_frame_count > 0 and i == arrival_frame_count:
                content.append("=== END BOWL ARRIVAL WINDOW | START BOWL RETURN WINDOW ===")
            try:
                image_data = base64.b64decode(base64_frame)
                image = Image.open(io.BytesIO(image_data))
                content.append(f"video={timestamp:.1f}s")
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
        """Return the sorted detections directly, as each service event is independent per customer."""
        if not detections:
            return []
        
        # Sort by video seated time
        return sorted(detections, key=lambda d: d.video_seated_time)

    def run_phase2(self, video_path: str, phase1_detections: List[Detection]) -> List[Detection]:
        """Run phase 2: Verification.
        
        Overlapping intervals from Phase 1 are merged into single clips.
        Uses the same cropped frames with PNG/Lanczos settings as Phase 1.
        """
        logger.info("PHASE 2: Verification")
        consolidated = self.consolidate_detections(phase1_detections)
        
        logger.info(f"Consolidated {len(phase1_detections)} Phase 1 detections into {len(consolidated)} clips for verification")
        
        verified = []
        
        WINDOW = 5  # seconds on each side of the event

        for idx, detection in enumerate(consolidated):
            logger.info(f"Verifying {idx+1}/{len(consolidated)}: Customer {detection.customer_id} at {detection.video_seated_time:.1f}s")

            # --- Window 1: bowl arrival ±WINDOW seconds ---
            arrival_clip = os.path.join(self.temp_dir, f"clip_{idx+1}_arrival.mp4")
            try:
                self.clip_video_segment(
                    video_path,
                    max(0.0, detection.video_seated_time - WINDOW),
                    detection.video_seated_time + WINDOW,
                    arrival_clip,
                )
                arrival_frames = self.extract_frames_phase2(arrival_clip)
            except Exception as e:
                logger.error(f"Failed to extract arrival window for detection {idx+1} (Customer {detection.customer_id}): {e}")
                arrival_frames = []

            # --- Window 2: bowl return ±WINDOW seconds ---
            return_clip = os.path.join(self.temp_dir, f"clip_{idx+1}_return.mp4")
            try:
                self.clip_video_segment(
                    video_path,
                    max(0.0, detection.video_serving_time - WINDOW),
                    detection.video_serving_time + WINDOW,
                    return_clip,
                )
                return_frames = self.extract_frames_phase2(return_clip)
            except Exception as e:
                logger.error(f"Failed to extract return window for detection {idx+1} (Customer {detection.customer_id}): {e}")
                return_frames = []

            if not arrival_frames and not return_frames:
                logger.error(f"No frames extracted for detection {idx+1} — skipping.")
                continue

            # Re-anchor timestamps to original video time so the model sees absolute times
            def _anchor(frames: List[Tuple[float, str]], clip_start: float) -> List[Tuple[float, str]]:
                return [(clip_start + t, b64) for t, b64 in frames]

            arrival_frames_abs = _anchor(arrival_frames, max(0.0, detection.video_seated_time - WINDOW))
            return_frames_abs  = _anchor(return_frames,  max(0.0, detection.video_serving_time - WINDOW))

            clip_frames = arrival_frames_abs + return_frames_abs
            context = f"Potential service event for customer: {detection.customer_id}"
            result = self.analyze_clip_phase2(
                clip_frames, context, idx + 1, detection,
                arrival_frame_count=len(arrival_frames_abs),
            )
            
            # Verification frame saving disabled to reduce disk usage
            
            # Save response
            with open(os.path.join(self.responses_dir, f"response_{idx+1}.json"), 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            if result.get("is_valid", False):
                verified_det = Detection(
                    customer_id=result.get("customer_id_verified") or detection.customer_id,
                    seated_time=result.get("verified_bowl_arrival_time") or detection.seated_time,
                    serving_time=result.get("verified_return_time") or detection.serving_time,
                    video_seated_time=detection.video_seated_time,
                    video_serving_time=detection.video_serving_time,
                    confidence=self._robust_float(result.get("confidence")),
                    description=result.get("description", ""),
                    phase=2,
                    appearance_description=result.get("appearance_details_verified"),
                    tracking_notes=result.get("tracking_consistency_notes"),
                    is_valid=True,
                    appearance_match_confirmed=result.get("appearance_match_confirmed"),
                    service_time_seconds=self._robust_float(result.get("eating_duration_seconds")),
                    reasoning=result.get("reasoning", ""),
                    seated_verified=result.get("return_event_confirmed"),
                    serving_verified=result.get("return_event_confirmed"),
                    bowl_color_confirmed=None,
                    customer_tracking_consistent=result.get("customer_tracking_consistent"),
                    timestamp_quality=result.get("timestamp_quality"),
                    bowl_completion_status=result.get("bowl_completion_status", "UNKNOWN"),
                    remaining_contents=result.get("remaining_contents"),
                )
                verified.append(verified_det)
                logger.info(f"[OK] VERIFIED: Customer={verified_det.customer_id}, Status={verified_det.bowl_completion_status}, Duration={verified_det.service_time_seconds}s")
            else:
                logger.info(f"[X] REJECTED: {result.get('reasoning')}")
        
        self.phase2_detections = verified
        
        # Capture and save the exact frames at verified start/end times
        self.capture_event_frames(video_path, verified, phase_label="phase2")
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
                self.clip_video_segment(video_path, detection.video_seated_time, detection.video_serving_time, clip_temp)
                f.write(f"file '{clip_temp.replace(os.sep, '/')}'\n")
        
        merged_path = os.path.join(self.config.output_dir, "merged_service_time_clips.mp4")
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
            "customer_id": [],
            "bowl_arrival_time_ocr": [],
            "return_time_ocr": [],
            "eating_duration_seconds": [],
            "bowl_completion_status": [],
            "remaining_contents": [],
            "confidence": [],
            "appearance": [],
            "description": [],
            "reasoning": [],
            "return_event_confirmed": [],
            "appearance_match_confirmed": [],
            "customer_tracking_consistent": [],
            "timestamp_quality": []
        }

        for idx, det in enumerate(detections, 1):
            data["customer_id"].append(det.customer_id)
            data["bowl_arrival_time_ocr"].append(det.seated_time)
            data["return_time_ocr"].append(det.serving_time)
            data["eating_duration_seconds"].append(det.service_time_seconds)
            data["bowl_completion_status"].append(det.bowl_completion_status)
            data["remaining_contents"].append(det.remaining_contents)
            data["confidence"].append(det.confidence)
            data["appearance"].append(det.appearance_description)
            data["description"].append(det.description)
            data["reasoning"].append(det.reasoning)
            data["return_event_confirmed"].append(det.seated_verified)
            data["appearance_match_confirmed"].append(det.appearance_match_confirmed)
            data["customer_tracking_consistent"].append(det.customer_tracking_consistent)
            data["timestamp_quality"].append(det.timestamp_quality)
        
        return pd.DataFrame(data)

    def save_results(self):
        """Save all analysis results."""
        output_path = os.path.join(self.config.output_dir, "service_time_analysis_results.json")
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
        token_output_path = os.path.join(self.config.output_dir, "service_time_token_usage.json")
        with open(token_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_usage, f, indent=2)
        logger.info(f"Token usage saved to: {token_output_path}")
        
        # Save CSV
        df = self.create_results_dataframe()
        csv_path = os.path.join(self.config.output_dir, "service_time_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV results saved to: {csv_path}")
        
        # Text summary
        summary_path = os.path.join(self.config.output_dir, "service_time_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"BOWL COMPLETION ANALYSIS SUMMARY\n{'='*50}\n")
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
            f.write("VERIFIED BOWL RETURN EVENTS:\n")
            f.write("-" * 50 + "\n")

            detections = self.phase2_detections if self.config.enable_phase2 else self.phase1_detections
            for idx, d in enumerate(detections, 1):
                f.write(f"\n{idx}. Customer ID: {d.customer_id}\n")
                f.write(f"   Bowl Arrival: {d.seated_time} | Return: {d.serving_time}\n")
                f.write(f"   Eating Duration: {d.service_time_seconds}s\n")
                f.write(f"   Completion Status: {d.bowl_completion_status}\n")
                f.write(f"   Remaining Contents: {d.remaining_contents or 'N/A'}\n")
                f.write(f"   Confidence: {d.confidence:.2f}\n")
                f.write(f"   Appearance: {d.appearance_description}\n")
                f.write(f"   Description: {d.description}\n")
        
        logger.info(f"Summary saved to: {summary_path}")

    # =========================================================================
    # Main Run Method
    # =========================================================================

    def run(self):
        """Main pipeline execution."""
        try:
            # Step 1: ROI Selection
            print("\n" + "="*60)
            print("CUSTOMER SERVICE TIME ANALYSIS PIPELINE")
            print("="*60)
            print(f"Video: {self.config.input_video_path}")
            print(f"Phase 1 Model: {self.config.phase1_model_name}")
            if self.config.enable_phase2:
                print(f"Phase 2 Model: {self.config.phase2_model_name}")
            print("="*60)
            
            if not self.roi:
                print("\nStep 1: Select the Region of Interest (ROI)")
                print("This should be the area containing the entrance and customer seating.")
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
    parser = argparse.ArgumentParser(description="Customer Service Time Analysis Pipeline")
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--fps", type=float, default=FPS, help="Frames per second to analyze")
    parser.add_argument("--no-phase2", action="store_true", help="Disable Phase 2 verification")
    args = parser.parse_args()
    
    # Determine video path
    video_path = args.video_path or DEFAULT_INPUT_VIDEO
    
    if not video_path:
        print("Error: No video path provided.")
        print("Usage: python serve_time.py <video_path>")
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
    pipeline = CustomerServicePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
