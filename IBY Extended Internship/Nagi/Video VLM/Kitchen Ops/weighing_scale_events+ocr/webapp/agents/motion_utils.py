"""
motion_utils.py — Mean Absolute Frame Difference (MAFD) pre-screening.

Batches whose ROI shows little motion (score < threshold) are idle periods
and can be skipped before calling the LLM, saving 30-50% of Phase 1 API
calls on quiet footage (DyToK arXiv 2512.06866).

Usage in each agent's run_phase1() batch loop:

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
from typing import List, Tuple

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
