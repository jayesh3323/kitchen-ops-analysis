"""
Timestamp OCR Module
---------------------
Extracts recording date and hour (HH) from a video frame using GPT-5-mini.
Used to compute real wall-clock timestamps for detected events using the formula:

  real_time = HH:MM:SS where
    total_seconds = event_seconds + recording_hour * 3600
    HH = INT(total_seconds / 3600)
    MM = INT(MOD(total_seconds, 3600) / 60)
    SS = MOD(total_seconds, 60)

This matches the Excel formula:
  =TEXT(INT((B2 + X*3600)/3600), "00") & ":" & TEXT(INT(MOD((B2 + X*3600), 3600)/60), "00") & ":" & TEXT(MOD((B2 + X*3600), 60), "00")
where B2 = event timestamp in seconds, X = recording_hour from OCR.
"""
import base64
import json
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TIMESTAMP_OCR_PROMPT = """You are analyzing a video recording frame that contains a timestamp overlay.

Your task is to extract:
1. The RECORDING DATE in YYYY-MM-DD format (e.g. "2026-01-25")
2. The RECORDING HOUR as an integer 0-23 (the HH part of the timestamp, in 24-hour format)

The timestamp is typically shown as an overlay on the video frame, often in a corner.
It may appear as: "2026-01-25 21:00:00" or "2026/01/25 21:00" or similar formats.

Return ONLY a JSON object with this exact structure:
{
  "recording_date": "YYYY-MM-DD",
  "recording_hour": <integer 0-23>,
  "raw_timestamp": "<the full timestamp text you found>",
  "confidence": <0.0 to 1.0>
}

If you cannot find a timestamp, return:
{
  "recording_date": null,
  "recording_hour": null,
  "raw_timestamp": null,
  "confidence": 0.0
}
"""


def extract_timestamp_from_frame(
    frame_base64: str,
    openai_client,
    model: str = "gpt-5-mini",
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Extract recording date and hour from a video frame using GPT-5-mini OCR.

    Args:
        frame_base64: Base64-encoded JPEG frame.
        openai_client: Initialized OpenAI client.
        model: Model to use for OCR.

    Returns:
        Tuple of (recording_date, recording_hour, raw_timestamp_text)
        e.g. ("2026-01-25", 21, "2026-01-25 21:00:00")
    """
    try:
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",", 1)[1]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": TIMESTAMP_OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                    }
                ]
            }],
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw.replace("```json", "", 1)
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        result = json.loads(raw)

        recording_date = result.get("recording_date")
        recording_hour = result.get("recording_hour")
        raw_timestamp = result.get("raw_timestamp")
        confidence = result.get("confidence", 0.0)

        logger.info(
            f"Timestamp OCR: date={recording_date}, hour={recording_hour}, "
            f"raw='{raw_timestamp}', confidence={confidence}"
        )

        # Validate
        if recording_hour is not None:
            recording_hour = int(recording_hour)
            if not (0 <= recording_hour <= 23):
                logger.warning(f"Invalid recording_hour: {recording_hour}, ignoring")
                recording_hour = None

        return recording_date, recording_hour, raw_timestamp

    except Exception as e:
        logger.error(f"Timestamp OCR failed: {e}")
        return None, None, None


def compute_real_timestamp(event_seconds: float, recording_hour: int) -> str:
    """
    Convert pipeline event timestamp (seconds from video start) to real wall-clock time.

    Formula (matches Excel):
      total = event_seconds + recording_hour * 3600
      HH = INT(total / 3600)
      MM = INT(MOD(total, 3600) / 60)
      SS = MOD(total, 60)

    Args:
        event_seconds: Timestamp from pipeline JSON (seconds from video start).
        recording_hour: HH value extracted from recording timestamp overlay.

    Returns:
        Wall-clock time string in "HH:MM:SS" format.
    """
    total_seconds = float(event_seconds) + recording_hour * 3600
    hh = int(total_seconds // 3600)
    mm = int((total_seconds % 3600) // 60)
    ss = int(total_seconds % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def compute_real_timestamps_for_results(
    detections: list,
    recording_hour: Optional[int],
    time_field: str = "start_time",
) -> list:
    """
    Add real_timestamp field to each detection dict using recording_hour.

    If recording_hour is None, real_timestamp is set to None.

    Args:
        detections: List of detection dicts with 'start_time' field.
        recording_hour: HH from recording timestamp OCR.
        time_field: Which dict key holds the video-seconds timestamp (default 'start_time').

    Returns:
        Same list with 'real_timestamp' added to each detection.
    """
    for det in detections:
        if recording_hour is not None:
            raw = det.get(time_field, 0) or 0
            # Plating detections use start_time as an OCR HH:MM:SS string;
            # fall back to video_start_time (seconds) when it's not numeric.
            if isinstance(raw, str) and ":" in raw:
                raw = det.get("video_start_time", 0) or 0
            start_time = float(raw)
            det["real_timestamp"] = compute_real_timestamp(start_time, recording_hour)
        else:
            det["real_timestamp"] = None
    return detections
