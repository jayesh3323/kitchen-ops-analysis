"""
Automatic ROI detection using a Vision Language Model (VLM).

Samples 3 frames from the video, sends each to Gemini 2.5 Pro with an
agent-specific prompt (including visual-content cues) that explains what
region to look for.  Optional KB reference images are prepended as
few-shot visual context.
Returns the best bounding box [x1, y1, x2, y2] in full-resolution pixel coords.
"""
import base64
import glob
import io
import json
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

import config as app_config

logger = logging.getLogger(__name__)

# Fractional positions in the video to sample (10%, 30%, 50%)
SAMPLE_POSITIONS = [0.10, 0.30, 0.50]

# =============================================================================
# Agent-specific VLM prompt context
# =============================================================================

_AGENT_CONTEXT = {
    "pork_weighing": (
        "This is a kitchen CCTV recording from a pork processing station or restaurant kitchen. "
        "Locate ALL weighing scales visible in the frame and return a single bounding box that covers every scale apparatus: "
        "the platform/tray, any item or bowl currently on it, and the digital readout together as one region. "
        "Include a generous margin of empty space around the outermost edges of the scale(s) so the full apparatus is never clipped."
    ),
    "plating_time": (
        "This is a kitchen CCTV recording of a ramen or noodle restaurant. "
        "Locate the table or counter where bowls are being assembled and prepared for service. "
        "Return a single bounding box that covers the ENTIRE table surface, including all bowls, "
        "ingredients, and tools on it. Include a generous margin around the table edges."
    ),
    "serve_time": (
        "This is a restaurant CCTV recording. "
        "Locate ONLY the kitchen pass-through counter — the narrow counter surface that physically "
        "separates the kitchen work area from the customer seating area. "
        "The kitchen side of this counter holds cooking equipment, containers, and stacked dishware. "
        "The customer side is the flat pass surface where completed bowls are placed before service. "
        "This is a NARROW BAND — typically a fraction of the full frame height. "
        "Return a bounding box that covers ONLY this counter band, including the kitchen equipment "
        "immediately behind it and the serving pass surface. "
        "Do NOT box the entire frame. Do NOT include the ceiling, walls, floor, or open dining area. "
        "In fisheye or wide-angle views the counter may run diagonally — follow the counter band only. "
        "The box must NOT exceed roughly half the image area."
    ),
    "noodle_rotation": (
        "This is a kitchen CCTV recording. "
        "Locate the noodle cooking station — around a pot, wok, or noodle basket — "
        "and return its bounding box. Include a generous margin around the station."
    ),
    "bowl_completion_rate": (
        "This is a restaurant CCTV recording, typically from an overhead or angled camera. "
"Locate ONLY the dining counter surface — the narrow counter or bar at which customers sit "
"and eat. This counter has condiment containers and napkin dispensers arranged on its surface, "
"and has seating positions directly adjacent to it. "
"Return a bounding box that covers ONLY the counter top surface and the immediate seating edge. "
"Do NOT include the kitchen, cooking equipment, open floor space, walls, or entrance. "
"Do NOT box the entire frame. "
"The box should tightly enclose the counter surface — it is a NARROW BAND, "
"typically covering well under half the image area."
),
}

# Visual content description for what should be SEEN inside the target ROI box.
# Used in the VLM prompt and in the KB few-shot introduction.
_AGENT_VISUAL_CUES = {
    "pork_weighing": (
        "The target region must contain ALL weighing scales visible in the frame as one bounding box. "
        "For each scale, include: the platform/tray where items are placed, any bowl or item on it, "
        "AND the digital readout panel that displays the weight. "
        "IMPORTANT: the digital display panel is often mounted to the SIDE or BACK of the platform — "
        "it may appear as a small rectangular LCD to the right of or behind the platform. "
        "The bbox MUST extend far enough to include every display panel. "
        "Add a clear margin of empty space on all sides around the outermost scale edges — "
        "do NOT crop tight to the scale body."
    ),
    "plating_time": (
        "Inside the target box you will see the full plating table surface: ramen or noodle bowls "
        "being assembled, ingredients being added, and staff hands working over the bowls. "
        "The entire table — from edge to edge — must be enclosed, not just the active bowl."
    ),
    "serve_time": (
        "Inside the target box you will see ONLY the kitchen counter band: "
"cooking equipment, pots, and containers on the kitchen side; "
"stacked dishware and serving supplies; "
"the flat pass surface where completed bowls are placed for pickup. "
"EXCLUDE: the entire dining floor, open seating area, walls, ceiling, and entrance. "
"EXCLUDE: anything more than one counter-width above or below the counter surface itself. "
"The box should be a horizontal or diagonal band — NOT a box covering most of the frame."
    ),
    "noodle_rotation": (
        "Inside the target box you will see the noodle cooking station: a pot, wok, or "
        "noodle basket with hot water or steam, and utensils or hands handling noodles."
    ),
    "bowl_completion_rate": (
        "Inside the target box you will see ONLY the dining counter surface: "
"condiment containers and napkin dispensers arranged along the counter top; "
"the flat counter surface where food bowls are placed for eating; "
"seating positions (empty spots) directly adjacent to the counter. "
"EXCLUDE: the kitchen, cooking or prep equipment, open floor space, walls, ceiling, "
"and any area more than one counter-width away from the counter surface. "
"The box should be a narrow band following the counter — NOT a box covering most of the frame."
    ),
}

# Per-agent bbox expansion margin (fraction of the detected bbox's own width/height).
# Larger value → more context added around the detected region.
# Small objects (scale) need large expansion; wide areas (counter) need smaller.
_AGENT_MARGIN = {
    "pork_weighing":  0.8,   # scale is small — expand generously to include display + platform
    "plating_time":   0,     # plating table is large — 0 margin to avoid spilling off-frame
    "serve_time":     0,     # seating area spans wide — 0 margin to keep focus on counter
    "noodle_rotation": 0.4,
    "bowl_completion_rate": 0, # covers two zones already — 0 margin to exclude customer area
}
_DEFAULT_MARGIN = 0

# Extra radius added to each detected display circle (fraction of the raw radius).
# 0.0 = no padding, 1.0 = 100% extra (double), 2.5 = 250% extra (3.5× raw radius).
# pork_weighing_compliance.py imports this — auto_roi.py is the single source of truth.
CIRCLE_RADIUS_MARGIN = 2.5

_DEFAULT_CONTEXT = (
    "This is a kitchen or restaurant CCTV recording. "
    "Identify the primary operational region relevant to food preparation or service activity."
)

_DEFAULT_VISUAL_CUE = (
    "Inside the target box you will see the main area of food preparation or service activity "
    "that is most relevant to the task being performed."
)

_VLM_PROMPT_TEMPLATE = (
    "{agent_context}\n\n"
    "Visual content that MUST be inside the bounding box:\n"
    "{visual_cue}\n\n"
    "Draw a box that fully encloses ALL the visual elements described above, with a generous margin of empty space on every side around the scale(s). Do NOT draw the tightest possible box — leave clear padding beyond the scale edges.\n"
    "A box covering less than 2% of the total image area is almost certainly wrong.\n\n"
    "Return ONLY a JSON object using NORMALIZED coordinates (fractions between 0.0 and 1.0,\n"
    "where 0.0 = left/top edge and 1.0 = right/bottom edge of the image).\n"
    "Do NOT return pixel values — return fractions of image width and height.\n\n"
    "MARGIN RULE:\n"
    "- If THE AGENT is 'pork_weighing', leave a GENEROUS margin (clear padding) around the scale edges.\n"
    "- If THE AGENT is 'plating_time', 'serve_time', or 'bowl_completion_rate', draw a TIGHT precision box \n"
    "  that covers the counter/table surface exactly without excessive empty space.\n\n"
    '{{"found": true, "x1": <0.0–1.0>, "y1": <0.0–1.0>, "x2": <0.0–1.0>, "y2": <0.0–1.0>, "confidence": <0.0–1.0>, "reasoning": "<one sentence>"}}\n\n'
    "If the relevant region is not visible in this frame, return:\n"
    '{{"found": false, "confidence": 0.0, "reasoning": "<why not found>"}}\n\n'
    "Coordinate rules:\n"
    "- All values must be between 0.0 and 1.0\n"
    "- x2 > x1, y2 > y1\n"
    "- (x2-x1)*(y2-y1) must be at least 0.02 (box must cover at least 2% of the image)"
)

_PORK_ROI_PROMPT = (
    "You are analyzing a single CCTV frame from a kitchen weighing station.\n"
    "Identify a single, tight bounding box (GREEN ROI) that encloses ONLY the weighing scale apparatus "
    "(the digital displays and the immediate platform/bowl-holder area).\n"
    "Do NOT include large amounts of surrounding empty counter space or other tools.\n\n"
    "Return ONLY a JSON object:\n"
    "{\n"
    '  "found": true,\n'
    '  "roi": {"x1": <0.0-1.0>, "y1": <0.0-1.0>, "x2": <0.0-1.0>, "y2": <0.0-1.0>},\n'
    '  "confidence": <0.0-1.0>,\n'
    '  "reasoning": "<one sentence>"\n'
    "}\n"
    "Use normalized fractions (0.0 to 1.0) relative to image width/height."
)

_PORK_DISPLAY_PROMPT_TEMPLATE = (
    "You are analyzing a CCTV frame from a kitchen weighing station.\n"
    "We have already identified the primary scale area (ROI) at: {{ \"x1\": {x1}, \"y1\": {y1}, \"x2\": {x2}, \"y2\": {y2} }}.\n\n"
    "Your task is to identify EACH digital readout screen (LCD/LED showing numeric weight) within or on this scale.\n"
    "CRITICAL: Look for small rectangular screens on the scale base, often with a green/backlit background.\n"
    "DO NOT confuse bowls, pork, or plates with the scale displays.\n\n"
    "Return ONLY a JSON object:\n"
    "{{\n"
    '  "displays": [\n'
    '    {{"x1": <0.0-1.0>, "y1": <0.0-1.0>, "x2": <0.0-1.0>, "y2": <0.0-1.0>}}, ...\n'
    '  ],\n'
    '  "reasoning": "<one sentence>"\n'
    "}}\n"
    "Coordinates must be normalized fractions of the FULL IMAGE width/height."
)


# =============================================================================
# Frame extraction helper
# =============================================================================

def _extract_frame_numpy(
    video_path: str,
    position_frac: float,
    rotation_angle: int = 0,
) -> Optional[np.ndarray]:
    """Extract a single frame at `position_frac` (0.0–1.0) of the video duration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None
        target = max(0, min(int(total * position_frac), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None

        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame
    finally:
        cap.release()


def _frame_to_base64(frame: np.ndarray) -> Tuple[str, int, int]:
    """Encode a numpy frame to base64 JPEG. Returns (b64_str, width, height)."""
    h, w = frame.shape[:2]
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return b64, w, h


# =============================================================================
# VLM detection
# =============================================================================

# =============================================================================
# Knowledge-base loader (folder-based)
# =============================================================================

def _load_kb_from_folder(agent: str, kb_dir: str) -> List[str]:
    """
    Load reference images for `agent` from `kb_dir/{agent}/`.
    Returns a list of base64-encoded JPEG/PNG strings (up to 3 images).
    Images may be plain frames or annotated frames (with a rectangle drawn).
    """
    # Try raw name first
    agent_dir = os.path.join(kb_dir, agent)
    if not os.path.isdir(agent_dir):
        # Handle common aliases before fallback
        _KB_ALIASES = {
            "bowl_completion": "bowl_completion_rate",
            "avg_serve_time": "serve_time"
        }
        agent_dir = os.path.join(kb_dir, _KB_ALIASES.get(agent, agent))
    
    if not os.path.isdir(agent_dir):
        # Fall back to canonical module name if available
        return []

    paths: List[str] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(agent_dir, pattern)))
    paths = sorted(paths)  # deterministic order, using all available reference images

    examples: List[str] = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                examples.append(base64.b64encode(f.read()).decode("utf-8"))
        except Exception as e:
            logger.warning(f"Could not load KB image {p}: {e}")
    return examples


def _b64_to_pil(b64_str: str) -> PILImage.Image:
    """Decode a base64-encoded image string to a PIL Image."""
    return PILImage.open(io.BytesIO(base64.b64decode(b64_str)))


def detect_roi_vlm(
    frame_base64: str,
    image_width: int,
    image_height: int,
    agent: str = "pork_weighing",
    kb_images: Optional[List[str]] = None,
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Use Gemini 2.5 Pro (or configured AUTO_ROI_VLM_MODEL) to locate the ROI.

    Args:
        frame_base64:  Base64-encoded JPEG frame.
        image_width:   Width of the original frame in pixels.
        image_height:  Height of the original frame in pixels.
        agent:         Agent name — determines what region to look for.
        kb_images:     Optional list of base64-encoded reference images loaded
                       from the KB folder.  Prepended as few-shot visual context.

    Returns:
        ((x1, y1, x2, y2), confidence, displays) or (None, 0.0, []) on failure/not-found.
    """
    if agent == "pork_weighing":
        return _detect_combined_pork_vlm(frame_base64, image_width, image_height)

    agent_context = _AGENT_CONTEXT.get(agent, _DEFAULT_CONTEXT)
    visual_cue = _AGENT_VISUAL_CUES.get(agent, _DEFAULT_VISUAL_CUE)
    prompt = _VLM_PROMPT_TEMPLATE.format(
        agent_context=agent_context,
        visual_cue=visual_cue,
    )

    try:
        import google.genai as genai
        client = genai.Client(api_key=app_config.GOOGLE_API_KEY)
        model = getattr(app_config, "AUTO_ROI_VLM_MODEL", "gemini-2.5-pro")

        # ── Build content list: optional few-shot KB images + query image ─────
        # Gemini accepts a flat list of strings (text) and PIL Images interleaved.
        contents = []

        if kb_images:
            contents.append(
                f"I will show you {len(kb_images)} reference image(s) with a green rectangle drawn over the correct target region.\n\n"
                f"Study what is INSIDE the green rectangle in each reference image — "
                f"that is the visual content you must locate in the new image:\n"
                f"{visual_cue}\n\n"
                f"The camera angle or store layout may differ — focus on the content, not the exact position or size shown in the reference. "
                f"Leave a generous margin of empty space around the target region — do NOT draw the tightest possible box."
            )
            for i, b64 in enumerate(kb_images, 1):
                contents.append(_b64_to_pil(b64))
            contents.append(
                "Now locate the same content in this new image and return its bounding box:"
            )

        contents.append(prompt)
        contents.append(_b64_to_pil(frame_base64))

        if kb_images:
            logger.info(f"VLM ROI [{agent}]: using {len(kb_images)} KB reference image(s)")

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

        if not result.get("found"):
            logger.info(f"VLM: region not found — {result.get('reasoning', '')}")
            return None, 0.0

        # Model should return normalized coords (0.0–1.0), but some models
        # return pixel values despite instructions.  Auto-detect which format
        # was used: if any value is clearly > 1.5, treat all as pixel coords.
        raw_x1 = float(result["x1"])
        raw_y1 = float(result["y1"])
        raw_x2 = float(result["x2"])
        raw_y2 = float(result["y2"])

        if max(raw_x1, raw_y1, raw_x2, raw_y2) > 1.5:
            # Model returned pixel coordinates — normalise them
            logger.info(
                f"VLM returned pixel coords ({raw_x1},{raw_y1})->({raw_x2},{raw_y2}); "
                f"normalising by {image_width}x{image_height}"
            )
            nx1 = raw_x1 / image_width
            ny1 = raw_y1 / image_height
            nx2 = raw_x2 / image_width
            ny2 = raw_y2 / image_height
        else:
            nx1, ny1, nx2, ny2 = raw_x1, raw_y1, raw_x2, raw_y2

        # Clamp to [0, 1]
        nx1, nx2 = max(0.0, nx1), min(1.0, nx2)
        ny1, ny2 = max(0.0, ny1), min(1.0, ny2)

        if nx2 <= nx1 or ny2 <= ny1:
            logger.warning(f"VLM returned invalid normalized bbox: ({nx1},{ny1})->({nx2},{ny2})")
            return None, 0.0

        norm_area = (nx2 - nx1) * (ny2 - ny1)
        if norm_area < 0.02:
            logger.warning(
                f"VLM bbox too small: covers {norm_area*100:.1f}% of image "
                f"(normalized: ({nx1:.3f},{ny1:.3f})->({nx2:.3f},{ny2:.3f})) — discarding"
            )
            return None, 0.0

        # Convert to pixel coordinates
        x1 = int(nx1 * image_width)
        y1 = int(ny1 * image_height)
        x2 = int(nx2 * image_width)
        y2 = int(ny2 * image_height)

        confidence = float(result.get("confidence", 0.7))
        logger.info(
            f"VLM ROI [{agent}]: normalized ({nx1:.3f},{ny1:.3f})->({nx2:.3f},{ny2:.3f}) "
            f"→ pixels ({x1},{y1})->({x2},{y2}) | area={norm_area*100:.1f}% | "
            f"conf={confidence:.2f} | {result.get('reasoning', '')}"
        )
        return (x1, y1, x2, y2), confidence, []

    except Exception as e:
        logger.error(f"VLM ROI detection failed: {type(e).__name__}: {e}", exc_info=True)
        return None, 0.0, []


def _detect_combined_pork_vlm(
    frame_base64: str,
    image_width: int,
    image_height: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], float, List[dict]]:
    """Sequential detection for pork weighing using Gemini 2.5 Pro (ROI then Displays)."""
    try:
        import google.genai as genai
        gemini_client = genai.Client(api_key=app_config.GOOGLE_API_KEY)
        model = getattr(app_config, "AUTO_ROI_VLM_MODEL", "gemini-2.5-pro")
        frame_pil = _b64_to_pil(frame_base64)

        # ── Step 1: Detect ROI ────────────────────────────────────────────────
        logger.info("Pork Weighing ROI Detection (Step 1/2)...")
        roi_response = gemini_client.models.generate_content(
            model=model,
            contents=[_PORK_ROI_PROMPT, frame_pil],
            config={"temperature": 0.0, "response_mime_type": "application/json"},
        )

        roi_raw = roi_response.text.strip() if roi_response.text else ""
        if not roi_raw:
            logger.warning("VLM Step 1 empty response from Gemini")
            return None, 0.0, []
        logger.info(f"VLM Step 1 raw response: {roi_raw[:500]}")
        # Strip markdown fences if model wraps output
        if roi_raw.startswith("```"):
            roi_raw = roi_raw.split("```")[1]
            if roi_raw.startswith("json"):
                roi_raw = roi_raw[4:]
        roi_result = json.loads(roi_raw)

        # Accept coordinates regardless of "found" flag — some model versions omit it.
        # Support both nested {"roi": {x1,y1,x2,y2}} and flat {x1,y1,x2,y2} formats.
        roi_data = roi_result.get("roi") or roi_result
        _has_coords = all(k in roi_data for k in ("x1", "y1", "x2", "y2"))
        if not _has_coords:
            logger.warning(f"VLM Step 1: no coordinates in response — {roi_result.get('reasoning', roi_raw[:200])}")
            return None, 0.0, []

        nx1, ny1, nx2, ny2 = roi_data["x1"], roi_data["y1"], roi_data["x2"], roi_data["y2"]

        nx1, nx2 = max(0.0, float(nx1)), min(1.0, float(nx2))
        ny1, ny2 = max(0.0, float(ny1)), min(1.0, float(ny2))

        if nx2 <= nx1 or ny2 <= ny1:
            logger.warning(f"VLM Step 1: degenerate bbox ({nx1},{ny1})->({nx2},{ny2})")
            return None, 0.0, []

        x1, y1 = int(nx1 * image_width), int(ny1 * image_height)
        x2, y2 = int(nx2 * image_width), int(ny2 * image_height)
        roi_bbox = (x1, y1, x2, y2)
        confidence = float(roi_result.get("confidence", 0.8))

        logger.info(f"VLM Step 1 ROI: ({x1},{y1})->({x2},{y2}) conf={confidence:.2f}")

        # ── Step 2: Detect Displays within ROI ──────────────────────────────
        logger.info("Pork Weighing Display Detection (Step 2/2)...")
        display_prompt = _PORK_DISPLAY_PROMPT_TEMPLATE.format(
            x1=round(nx1, 3), y1=round(ny1, 3), x2=round(nx2, 3), y2=round(ny2, 3)
        )

        disp_response = gemini_client.models.generate_content(
            model=model,
            contents=[display_prompt, frame_pil],
            config={"temperature": 0.0, "response_mime_type": "application/json"},
        )

        disp_raw = disp_response.text.strip() if disp_response.text else "{}"
        if disp_raw.startswith("```"):
            disp_raw = disp_raw.split("```")[1]
            if disp_raw.startswith("json"):
                disp_raw = disp_raw[4:]
        disp_result = json.loads(disp_raw) if disp_raw else {}
        displays = disp_result.get("displays", [])

        # Convert display coords to pixels (Step 2 returns full-frame norm coords in this implementation)
        final_displays = []
        for d in displays:
            dx1, dy1, dx2, dy2 = d["x1"], d["y1"], d["x2"], d["y2"]
            # Convert norm to pixels
            px1, py1 = int(dx1 * image_width), int(dy1 * image_height)
            px2, py2 = int(dx2 * image_width), int(dy2 * image_height)

            # Sanity check: is it inside the detected ROI (plus 5% margin)?
            mx = (x2 - x1) * 0.05
            my = (y2 - y1) * 0.05
            if (x1 - mx <= px1 <= x2 + mx and y1 - my <= py1 <= y2 + my):
                final_displays.append({"x1": px1, "y1": py1, "x2": px2, "y2": py2})

        logger.info(f"VLM Step 2: {len(final_displays)} displays found (filtered from {len(displays)})")
        return roi_bbox, confidence, final_displays

    except Exception as e:
        logger.error(f"Sequential VLM detection failed: {e}", exc_info=True)
        return None, 0.0, []


# =============================================================================
# Annotation helper
# =============================================================================

def _draw_roi_annotation(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    confidence: float,
    displays: List[dict] = None,
) -> bytes:
    """Draw the detected ROI rectangle and optional display circles on the frame."""
    annotated = frame.copy()
    x1, y1, x2, y2 = bbox

    # 1. Draw Green ROI Box
    if confidence >= 0.70:
        color = (0, 200, 100)
    elif confidence >= 0.40:
        color = (0, 165, 255)
    else:
        color = (50, 50, 220)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    # 2. Draw Red Display Circles
    if displays:
        for d in displays:
            dx1, dy1, dx2, dy2 = d["x1"], d["y1"], d["x2"], d["y2"]
            cx = (dx1 + dx2) // 2
            cy = (dy1 + dy2) // 2
            r = max(dx2 - dx1, dy2 - dy1) // 2
            r = int(r * (1.0 + CIRCLE_RADIUS_MARGIN))
            cv2.circle(annotated, (cx, cy), r, (0, 0, 255), 2)

    label = f"AI {int(confidence * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.0, frame.shape[1] / 1280))
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad = 4
    # Keep label inside frame if the box is near the top
    label_y_top = max(th + baseline + pad * 2, y1)
    cv2.rectangle(
        annotated,
        (x1, label_y_top - th - baseline - pad * 2),
        (x1 + tw + pad * 2, label_y_top),
        color,
        -1,
    )
    cv2.putText(
        annotated,
        label,
        (x1 + pad, label_y_top - baseline - pad),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# =============================================================================
# Main orchestrator
# =============================================================================

def auto_detect_roi(
    video_path: str,
    rotation_angle: int = 0,
    agent: str = "pork_weighing",
    kb_dir: Optional[str] = None,
) -> dict:
    """
    Sample 3 frames from the video and ask the VLM to locate the ROI in each.
    Returns the result with the highest confidence.

    Args:
        video_path:     Path to the video file.
        rotation_angle: Rotation to apply before analysis (0/90/180/270).
        agent:          Agent name — drives what the VLM is told to look for.
        kb_dir:         Path to the root KB folder.  If provided, images from
                        `kb_dir/{agent}/` are passed as few-shot reference images.

    Returns:
        {
            "roi": [x1, y1, x2, y2] or None,
            "displays": list or None,
            "confidence": float,
            "method": "vlm" | "failed",
            "annotated_frame": bytes (JPEG) or None,
            "frame_w": int,
            "frame_h": int,
        }
    """
    # ── Normalize agent name ──────────────────────────────────────────────────
    _ALIASES = {
        "avg_serve_time": "serve_time",
        "bowl_completion": "bowl_completion_rate",
    }
    agent = _ALIASES.get(agent, agent)

    _failed = {
        "roi": None, "displays": [], "confidence": 0.0, "method": "failed",
        "annotated_frame": None, "frame_w": 0, "frame_h": 0,
    }

    try:
        # ── Load KB reference images for this agent ───────────────────────────
        kb_images: Optional[List[str]] = None
        if kb_dir:
            kb_images = _load_kb_from_folder(agent, kb_dir) or None

        # ── Sample frames ─────────────────────────────────────────────────────
        frames = []
        for pos in SAMPLE_POSITIONS:
            frame = _extract_frame_numpy(video_path, pos, rotation_angle)
            if frame is not None:
                frames.append(frame)

        if not frames:
            logger.warning("Could not extract any frames from video")
            return _failed

        logger.info(
            f"Auto-detecting ROI for agent='{agent}' across {len(frames)} frames"
            + (f" with {len(kb_images)} KB reference image(s)" if kb_images else "")
        )

        # ── Run VLM on each frame, collect results ────────────────────────────
        results = []  # list of (bbox, confidence, frame, w, h, displays)
        for i, frame in enumerate(frames):
            b64, w, h = _frame_to_base64(frame)
            bbox, conf, displays = detect_roi_vlm(b64, w, h, agent=agent, kb_images=kb_images)
            if bbox is not None:
                results.append((bbox, conf, frame, w, h, displays))
                logger.info(f"Frame {i+1}/{len(frames)}: found ROI conf={conf:.2f}")
            else:
                logger.info(f"Frame {i+1}/{len(frames)}: no ROI found")

        if not results:
            logger.warning(f"VLM found no ROI in any of the {len(frames)} sampled frames")
            return _failed

        # ── Pick the result with the highest confidence ───────────────────────
        best_bbox, best_conf, best_frame, best_w, best_h, best_displays = max(results, key=lambda r: r[1])
        x1, y1, x2, y2 = best_bbox

        # ── Expand bbox with per-agent margins (fraction of bbox dimension) ───
        margin = _AGENT_MARGIN.get(agent, _DEFAULT_MARGIN)
        bw = x2 - x1
        bh = y2 - y1
        x1 = max(0,       x1 - int(bw * margin))   # left
        x2 = min(best_w,  x2 + int(bw * margin))   # right
        y1 = max(0,       y1 - int(bh * margin))   # top
        y2 = min(best_h,  y2 + int(bh * margin))   # bottom

        logger.info(
            f"Best ROI: ({x1},{y1})->({x2},{y2}) conf={best_conf:.2f} "
            f"(from {len(results)}/{len(frames)} frames that found a region)"
        )

        annotated_bytes = _draw_roi_annotation(best_frame, (x1, y1, x2, y2), best_conf, best_displays)

        return {
            "roi": [x1, y1, x2, y2],
            "displays": best_displays,
            "confidence": round(best_conf, 3),
            "method": "vlm",
            "annotated_frame": annotated_bytes,
            "frame_w": best_w,
            "frame_h": best_h,
        }

    except Exception as e:
        logger.error(f"auto_detect_roi error: {e}", exc_info=True)
        return _failed
