"""
motion_utils.py — Motion utilities for VLM frame pre-processing.

Contains:
  - compute_batch_mafd(): MAFD pre-screening to skip idle batches before LLM calls
  - apply_optical_flow_overlay(): Farneback dense flow HSV colour overlay for
    explicit motion-direction cues visible to the VLM

Optical flow overlay usage in each agent's extract_frames() loop:

    from motion_utils import apply_optical_flow_overlay

    prev_gray_for_flow = None
    ...
    prepared = self.prepare_frame_for_analysis(frame)
    if self.config.optical_flow_overlay:
        prepared, prev_gray_for_flow = apply_optical_flow_overlay(prepared, prev_gray_for_flow)
    base64_frame = self._compress_frame(prepared, ...)

MAFD pre-screening usage in each agent's run_phase1() batch loop:

    from motion_utils import compute_batch_mafd

    for batch in batches:
        if config.motion_threshold > 0:
            score = compute_batch_mafd(batch.frames)
            if score < config.motion_threshold:
                logger.info(f"Batch {batch.batch_id} skipped — MAFD {score:.1f}")
                continue
        result = self.analyze_batch_phase1(batch, context)
"""
import base64
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_batch_mafd(
    batch_frames: List[Tuple[float, str]],
    n_samples: int = 3,
) -> float:
    """
    Compute mean absolute frame difference across sampled frames in a batch.

    Decodes at most *n_samples* evenly-spaced frames from the batch's
    base64-encoded images, converts them to grayscale, and computes the
    mean absolute pixel difference between consecutive samples.

    Args:
        batch_frames: List of (timestamp_s, base64_image_str) tuples as
                      stored in FrameBatch.frames.
        n_samples:    How many frames to sample.  3 is enough for a
                      reliable idle/active decision at low decode cost.

    Returns:
        Float in [0, 255].  Higher = more motion.
        Returns 255.0 (never skip) when computation fails or the batch
        has fewer than 2 frames.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.debug("motion_utils: cv2/numpy not available — skipping MAFD check")
        return 255.0

    if len(batch_frames) < 2:
        return 255.0

    total = len(batch_frames)
    if total <= n_samples:
        indices = list(range(total))
    else:
        indices = [
            int(round(i * (total - 1) / (n_samples - 1)))
            for i in range(n_samples)
        ]

    grays: List["np.ndarray"] = []
    for idx in indices:
        b64 = batch_frames[idx][1]
        try:
            # Strip data-URL prefix ("data:image/jpeg;base64,…") if present
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            raw = base64.b64decode(b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                grays.append(img.astype(np.float32))
        except Exception as exc:
            logger.debug(f"motion_utils: frame decode failed — {exc}")

    if len(grays) < 2:
        return 255.0

    h, w = grays[0].shape
    diffs: List[float] = []
    for j in range(1, len(grays)):
        g = grays[j]
        if g.shape != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        diffs.append(float(np.mean(np.abs(grays[0] - g))))

    return float(np.mean(diffs))


def apply_optical_flow_overlay(
    frame_bgr: "np.ndarray",
    prev_gray: Optional["np.ndarray"],
    alpha: float = 0.45,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Compute Farneback dense optical flow between *prev_gray* and *frame_bgr*,
    encode motion direction as HSV hue and magnitude as value, then alpha-blend
    the resulting colour map onto *frame_bgr*.

    The blended frame gives the VLM explicit colour-coded arrows so it can read
    rotation direction (e.g. clockwise noodle stirring → consistent hue swirl).

    HSV encoding:
      - Hue   → flow direction (0–360° mapped to 0–180 in OpenCV HSV)
      - Saturation → 255 (full)
      - Value → normalised magnitude (0 = no motion, 255 = peak motion)

    Args:
        frame_bgr:  Current frame as a BGR numpy array (HxWx3, uint8).
        prev_gray:  Grayscale previous frame (HxW, uint8), or None for the
                    first frame (overlay is skipped; original returned as-is).
        alpha:      Blend weight for the flow layer [0, 1].  0.45 keeps the
                    original detail clearly visible while making motion readable.

    Returns:
        (overlaid_bgr, curr_gray) — blended frame and the current grayscale
        frame ready to pass as *prev_gray* on the next call.
    """
    try:
        import cv2
        import numpy as np

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # First frame — nothing to diff against; return original unchanged.
            return frame_bgr, curr_gray

        # Ensure shapes match (frame might have been resized after preparation).
        pg = prev_gray
        if pg.shape != curr_gray.shape:
            pg = cv2.resize(pg, (curr_gray.shape[1], curr_gray.shape[0]), interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(
            pg, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Clip at 99th percentile to prevent one bright pixel dominating the scale.
        max_mag = float(np.percentile(mag, 99))
        if max_mag < 0.5:
            # Essentially no motion — skip overlay so frame stays unmodified.
            return frame_bgr, curr_gray

        hsv = np.zeros_like(frame_bgr)
        hsv[..., 1] = 255                                                    # full saturation
        hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)         # direction → hue
        hsv[..., 2] = np.clip(mag / max_mag * 255.0, 0, 255).astype(np.uint8)  # magnitude → value

        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, flow_bgr, alpha, 0)
        return blended, curr_gray

    except Exception as exc:
        logger.debug(f"motion_utils: optical flow overlay failed — {exc}")
        return frame_bgr, (prev_gray if prev_gray is not None else frame_bgr)  # type: ignore[return-value]
