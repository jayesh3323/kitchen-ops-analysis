"""
Headless adapter for analysis pipelines.
Wraps the existing pipelines for server-side execution without interactive GUI.
Reports progress via callback.
Langfuse tracing (v3 SDK) is enabled via langfuse_manager.
"""
import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import asdict

# Add parent directory and agents/ to path so we can import the pipeline scripts.
# Local dev:    parent dir (../../) contains pork_weighing_analysis.py etc.
# HF Spaces:    agents/ dir (inside the Docker image) contains them.
_parent_dir = str(Path(__file__).parent.parent)
_agents_dir = str(Path(__file__).parent / "agents")
# agents/ must take priority over parent dir — insert agents/ at front,
# append parent dir at the back so it never shadows agents/ modules.
if _agents_dir not in sys.path:
    sys.path.insert(0, _agents_dir)
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)

from pork_weighing_compliance import PorkWeighingPipeline, PipelineConfig as PorkConfig
from plating_time import RamenPlatingPipeline, PipelineConfig as PlatingConfig
from avg_serve_time import CustomerServicePipeline, PipelineConfig as ServeTimeConfig
from bowl_completion_rate import BowlCompletionPipeline, PipelineConfig as BowlConfig
from noodle_rotation_compliance import NoodleRotationPipeline, PipelineConfig as NoodleConfig
import config as app_config
from langfuse_manager import start_trace, start_child_span, end_span, flush as langfuse_flush

logger = logging.getLogger(__name__)


def _flush_langfuse_async():
    """Fire-and-forget Langfuse flush so it never blocks job completion."""
    import threading
    t = threading.Thread(target=langfuse_flush, daemon=True, name="langfuse-flush")
    t.start()


def run_pipeline_headless(
    video_path: str,
    output_dir: str,
    roi_coords: Optional[Tuple[int, int, int, int]] = None,
    agent: str = "pork_weighing",
    job_config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    job_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the analysis pipeline in headless mode (no GUI).
    Creates a Langfuse trace for the entire pipeline run.
    """

    def report(status: str, message: str):
        logger.info(f"[{status.upper()}] {message}")
        if progress_callback:
            progress_callback(status, message)

    # ── Normalize agent name to canonical form ─────────────────────────────
    _AGENT_ALIASES = {
        "avg_serve_time": "serve_time",
        "bowl_completion": "bowl_completion_rate",
    }
    agent = _AGENT_ALIASES.get(agent, agent)

    report("processing", f"Initializing {agent} pipeline...")
    logger.info(f"[ROUTING] Resolved agent name: '{agent}'")

    if job_config is None:
        job_config = {}

    # Load per-agent defaults so each task uses its own canonical parameters.
    ad = app_config.get_agent_defaults(agent)

    fps = job_config.get("fps", ad.get("AGENT_FPS", app_config.FPS))
    enable_phase2 = job_config.get("enable_phase2", app_config.ENABLE_PHASE2)

    # Extract advanced kwargs — job_config values take priority over agent defaults.
    kwargs = {}
    kwargs["confidence_threshold"]    = float(job_config.get("confidence_threshold",    ad.get("AGENT_CONFIDENCE_THRESHOLD",    app_config.CONFIDENCE_THRESHOLD)))
    kwargs["max_batch_size_mb"]       = float(job_config.get("max_batch_size_mb",      ad.get("AGENT_MAX_BATCH_SIZE_MB",       app_config.MAX_BATCH_SIZE_MB)))
    kwargs["clip_buffer_seconds"]     = int(job_config.get("clip_buffer_seconds",      ad.get("AGENT_CLIP_BUFFER_SECONDS",     app_config.CLIP_BUFFER_SECONDS)))
    kwargs["max_frames_per_batch"]    = int(job_config.get("max_frames_per_batch",     ad.get("AGENT_MAX_FRAMES_PER_BATCH",    app_config.MAX_FRAMES_PER_BATCH)))
    kwargs["batch_overlap_frames"]    = int(job_config.get("batch_overlap_frames",     ad.get("AGENT_BATCH_OVERLAP_FRAMES",    app_config.BATCH_OVERLAP_FRAMES)))
    kwargs["image_quality"]           = int(job_config.get("image_quality",            ad.get("AGENT_IMAGE_QUALITY",           app_config.IMAGE_QUALITY)))
    kwargs["image_upscale_factor"]    = float(job_config.get("image_upscale_factor",   ad.get("AGENT_IMAGE_UPSCALE_FACTOR",    app_config.IMAGE_UPSCALE_FACTOR)))
    kwargs["rotation_angle"]          = int(job_config.get("rotation_angle",           ad.get("AGENT_ROTATION_ANGLE",          app_config.ROTATION_ANGLE)))
    kwargs["phase1_model_name"]       = job_config.get("phase1_model_name",            ad.get("AGENT_PHASE1_MODEL_NAME",       app_config.PHASE1_MODEL_NAME))
    kwargs["phase2_model_name"]       = job_config.get("phase2_model_name",            ad.get("AGENT_PHASE2_MODEL_NAME",       app_config.PHASE2_MODEL_NAME))
    kwargs["image_interpolation"]     = job_config.get("image_interpolation",          ad.get("AGENT_IMAGE_INTERPOLATION",     app_config.IMAGE_INTERPOLATION))
    kwargs["enable_cropping"]         = job_config.get("enable_cropping",              ad.get("AGENT_ENABLE_CROPPING",         app_config.ENABLE_CROPPING))
    kwargs["image_format"]            = job_config.get("image_format",                 ad.get("AGENT_IMAGE_FORMAT",            app_config.IMAGE_FORMAT))
    kwargs["phase2_image_format"]     = job_config.get("phase2_image_format",          ad.get("AGENT_PHASE2_IMAGE_FORMAT",     app_config.PHASE2_IMAGE_FORMAT))
    kwargs["image_target_resolution"] = job_config.get("image_target_resolution",      ad.get("AGENT_IMAGE_TARGET_RESOLUTION", app_config.IMAGE_TARGET_RESOLUTION))

    # ── Create Langfuse root span (trace) for this pipeline run ────────────
    # session_id groups all jobs of the same task type into one "task folder"
    # in Langfuse.  The trace name is the job ID so each job is its own entry
    # rooted under that folder.
    trace = start_trace(
        name=f"job-{job_id}" if job_id else f"pipeline-{agent}",
        metadata={
            "agent": agent,
            "video": Path(video_path).name,
            "fps": fps,
            "enable_phase2": enable_phase2,
            "roi": list(roi_coords) if roi_coords else None,
            "job_id": job_id,
            **{k: str(v) for k, v in kwargs.items()},
        },
        session_id=agent,
        tags=[agent, "pipeline"],
    )

    if agent == "plating_time":
        pipeline_config = PlatingConfig(
            input_video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            enable_phase2=enable_phase2,
            roi=roi_coords,
            **kwargs
        )
        pipeline = RamenPlatingPipeline(pipeline_config)
    elif agent == "serve_time":
        # ServeTimeConfig has no image_quality or max_frames_per_batch fields
        serve_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ("image_quality", "max_frames_per_batch")}
        pipeline_config = ServeTimeConfig(
            input_video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            enable_phase2=enable_phase2,
            roi=roi_coords,
            **serve_kwargs
        )
        pipeline = CustomerServicePipeline(pipeline_config)
    elif agent == "bowl_completion_rate":
        pipeline_config = BowlConfig(
            input_video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            enable_phase2=enable_phase2,
            roi=roi_coords,
            recording_hour=job_config.get("recording_hour"),
            **kwargs
        )
        pipeline = BowlCompletionPipeline(pipeline_config)
    elif agent == "noodle_rotation":
        pipeline_config = NoodleConfig(
            input_video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            enable_phase2=enable_phase2,
            roi=roi_coords,
            **kwargs
        )
        pipeline = NoodleRotationPipeline(pipeline_config)
    elif agent == "pork_weighing":
        pipeline_config = PorkConfig(
            input_video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            enable_phase2=enable_phase2,
            roi=roi_coords,
            **kwargs
        )
        pipeline = PorkWeighingPipeline(pipeline_config)
    else:
        logger.error(f"[ROUTING] Unknown agent '{agent}' — falling back to pork_weighing. "
                     f"This is likely a bug. Valid agents: pork_weighing, plating_time, "
                     f"serve_time, bowl_completion_rate, noodle_rotation")
        pipeline_config = PorkConfig(
            input_video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            enable_phase2=enable_phase2,
            roi=roi_coords,
            **kwargs
        )
        pipeline = PorkWeighingPipeline(pipeline_config)

    # Override ROI so it skips interactive selection
    if roi_coords:
        pipeline.roi = roi_coords
        logger.info(f"Using provided ROI: {roi_coords}")

    # For pork_weighing: save CLAHE-enhanced preview frames before Phase 1
    if agent == "pork_weighing" and hasattr(pipeline, "save_clahe_preview_frames"):
        try:
            report("processing", "Saving CLAHE preview frames...")
            pipeline.save_clahe_preview_frames(video_path)
        except Exception as _clahe_err:
            logger.warning(f"CLAHE preview frames failed (non-fatal): {_clahe_err}")

    results = {
        "phase1_detections": [],
        "phase2_detections": [],
        "phase1_count": 0,
        "phase2_count": 0,
        "total_tokens": 0,
        "token_usage": {},
    }

    start_time = time.time()

    try:
        # =====================================================================
        # Phase 1: Event Detection
        # =====================================================================
        report("phase1", "Running Phase 1: Event Detection + Initial OCR...")

        # Log Phase 1 as a child span in Langfuse
        phase1_span = start_child_span(
            trace,
            name="phase1-detection",
            metadata={"model": pipeline_config.phase1_model_name},
        )

        phase1_detections = pipeline.run_phase1(video_path)
        results["phase1_detections"] = [asdict(d) for d in phase1_detections]
        results["phase1_count"] = len(phase1_detections)

        phase1_raw = pipeline.token_usage.get("phase1_raw_responses", [])
        end_span(
            phase1_span,
            metadata={
                "detections_found": len(phase1_detections),
                "phase1_tokens": pipeline.token_usage.get("total_phase1_tokens", 0),
            },
            output="\n\n".join(phase1_raw) if phase1_raw else None,
        )

        report("phase1", f"Phase 1 complete: {len(phase1_detections)} detections found")

        # =====================================================================
        # Phase 2: Verification
        # =====================================================================
        if pipeline_config.enable_phase2 and phase1_detections:
            report("phase2", "Running Phase 2: Verification with Gemini...")

            phase2_span = start_child_span(
                trace,
                name="phase2-verification",
                metadata={"model": pipeline_config.phase2_model_name},
            )

            phase2_detections = pipeline.run_phase2(video_path, phase1_detections)
            results["phase2_detections"] = [asdict(d) for d in phase2_detections]
            results["phase2_count"] = len(phase2_detections)

            phase2_raw = pipeline.token_usage.get("phase2_raw_responses", [])
            end_span(
                phase2_span,
                metadata={
                    "verified_events": len(phase2_detections),
                    "phase2_tokens": pipeline.token_usage.get("total_phase2_tokens", 0),
                },
                output="\n\n".join(phase2_raw) if phase2_raw else None,
            )

            # Create merged video
            try:
                pipeline.create_merged_video(video_path)
            except Exception as e:
                logger.warning(f"Failed to create merged video: {e}")

            report("phase2", f"Phase 2 complete: {len(phase2_detections)} verified events")

        elif not phase1_detections:
            report("phase2", "No Phase 1 detections found. Skipping Phase 2.")
            pipeline.phase2_detections = []
        else:
            report("processing", "Single-phase mode: Skipping Phase 2.")
            pipeline.phase2_detections = phase1_detections

        # =====================================================================
        # Save results
        # =====================================================================
        report("processing", "Saving results...")
        pipeline.save_results()

        # Collect token usage
        results["token_usage"] = pipeline.token_usage
        results["total_tokens"] = pipeline.token_usage.get("total_tokens", 0)

        elapsed = time.time() - start_time

        # Close the root span with final summary
        end_span(
            trace,
            metadata={
                "status": "completed",
                "phase1_count": results["phase1_count"],
                "phase2_count": results["phase2_count"],
                "total_tokens": results["total_tokens"],
                "elapsed_seconds": round(elapsed, 1),
            },
        )

        report("completed", "Analysis complete!")
        _flush_langfuse_async()
        return results

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)

        # Log error to Langfuse root span
        end_span(
            trace,
            metadata={
                "status": "failed",
                "error": str(e)[:500],
                "elapsed_seconds": round(time.time() - start_time, 1),
            },
        )

        _flush_langfuse_async()
        raise
    finally:
        pipeline.cleanup()


def extract_frame_for_roi(video_path: str, frame_index: int = 30) -> Optional[bytes]:
    """
    Extract a single frame from a video file for ROI / timestamp region selection.
    Returns the frame as JPEG bytes, or None on failure.
    """
    import cv2

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = min(frame_index, max(0, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Apply rotation if configured
        rotation = app_config.ROTATION_ANGLE
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    except Exception as e:
        logger.error(f"Error extracting frame: {e}")
        return None
