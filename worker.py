"""
Background worker for video processing.
Uses a simple SQLite-based task queue with threading — no Redis/Celery required.
Polls the database for queued jobs and processes them sequentially.
Langfuse tracing is enabled via pipeline_adapter → langfuse_manager.
"""
import os
import sys
import json
import logging
import threading
import signal
import base64
from datetime import datetime, timezone
from pathlib import Path

# Add webapp directory to path
_webapp_dir = Path(__file__).parent
if str(_webapp_dir) not in sys.path:
    sys.path.insert(0, str(_webapp_dir))

import config as app_config
from database import init_db, SessionLocal, Job, Result, TokenUsage

logger = logging.getLogger(__name__)

# Global flag to control worker loop
_shutdown_flag = threading.Event()


def _update_job(session, job_id: int, **kwargs):
    """Update job fields. In Firebase mode uses a write-only path (no read)."""
    if hasattr(session, "direct_update_job"):
        # Firebase mode: update Firestore fields directly — no prior read needed
        session.direct_update_job(job_id, **kwargs)
        return None
    # SQLite mode: read-then-write
    job = session.query(Job).filter(Job.id == job_id).first()
    if job:
        for key, value in kwargs.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)
        session.commit()
    return job


def process_single_job(job_id: int):
    """
    Process a single video analysis job.
    This runs the full pipeline (Phase 1 + Phase 2) and saves results.
    """
    session = SessionLocal()

    try:
        # Load job from database
        job = session.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database.")
            return

        video_path = job.video_path

        if not os.path.exists(video_path):
            _update_job(session, job_id,
                        status="failed",
                        error_message=f"Video file not found: {video_path}")
            return

        # Parse ROI coordinates
        roi_coords = None
        if job.roi_coords:
            try:
                coords = json.loads(job.roi_coords)
                roi_coords = tuple(coords)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid ROI coords for job {job_id}: {job.roi_coords}")

        # Parse timestamp region coordinates (for OCR crop)
        timestamp_region = None
        if job.timestamp_region_coords:
            try:
                coords = json.loads(job.timestamp_region_coords)
                timestamp_region = tuple(coords)  # (x1, y1, x2, y2)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid timestamp region for job {job_id}")

        # Parse config
        job_config = {}
        if job.config_json:
            try:
                job_config = json.loads(job.config_json)
            except json.JSONDecodeError:
                pass

        # Create output directory
        output_dir = os.path.join(
            app_config.RESULTS_DIR,
            f"job_{job_id}_{Path(job.video_name).stem}"
        )
        os.makedirs(output_dir, exist_ok=True)

        _update_job(session, job_id,
                    status="processing",
                    output_dir=output_dir,
                    progress_message="Initializing pipeline...")

        # ── Timestamp OCR (FIRST — on raw unrotated frame) ─────────────────
        # Always runs before rotation or ROI detection so coordinate spaces
        # never conflict. The camera timestamp overlay is horizontal in the
        # original frame regardless of how the video is rotated for analysis.
        recording_date = job.recording_date
        recording_hour = job.recording_hour

        if recording_date is None or recording_hour is None:
            _update_job(session, job_id,
                        progress_message="Extracting recording timestamp via OCR...")
            try:
                from timestamp_ocr import extract_timestamp_from_frame
                from openai import OpenAI
                import cv2
                import numpy as np

                # Extract the raw unrotated frame — no rotation applied here.
                cap_ts = cv2.VideoCapture(video_path)
                raw_frame = None
                if cap_ts.isOpened():
                    total_ts = int(cap_ts.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap_ts.set(cv2.CAP_PROP_POS_FRAMES, min(30, max(0, total_ts - 1)))
                    ret_ts, raw_frame = cap_ts.read()
                    cap_ts.release()

                if raw_frame is not None:
                    h, w = raw_frame.shape[:2]
                    if timestamp_region:
                        # User drew a region on the 0° frame — crop precisely.
                        x1, y1, x2, y2 = timestamp_region
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        ocr_frame = raw_frame[y1:y2, x1:x2]
                    else:
                        # No region selected — use the top band where CCTV
                        # timestamp overlays typically appear.
                        ocr_frame = raw_frame[:min(200, h), :]

                    _, buf = cv2.imencode(".jpg", ocr_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

                    ocr_client = OpenAI(api_key=app_config.OPENAI_API_KEY)
                    rec_date, rec_hour, raw_ts = extract_timestamp_from_frame(
                        frame_b64, ocr_client, model=app_config.PHASE1_MODEL_NAME
                    )
                    if rec_date or rec_hour is not None:
                        recording_date = rec_date
                        recording_hour = rec_hour
                        _update_job(session, job_id,
                                    recording_date=recording_date,
                                    recording_hour=recording_hour,
                                    progress_message=f"Timestamp detected: {raw_ts or 'unknown'}")
                        logger.info(f"Timestamp OCR: date={recording_date}, hour={recording_hour}, raw='{raw_ts}'")
                    else:
                        logger.warning("Timestamp OCR returned no result — timestamps will be relative.")
            except Exception as ts_err:
                logger.warning(f"Timestamp OCR step failed (non-fatal): {ts_err}")

        # Progress callback — updates job status in DB
        def progress_callback(status: str, message: str):
            try:
                _update_job(session, job_id,
                            status=status,
                            progress_message=message)
            except Exception as e:
                logger.warning(f"Failed to update progress: {e}")
                try:
                    session.rollback()
                except Exception:
                    pass

        # Inject recording_hour into job_config so pipeline adapters can use it
        if recording_hour is not None:
            job_config["recording_hour"] = recording_hour

        # Run the pipeline
        from pipeline_adapter import run_pipeline_headless

        resolved_agent = job.agent or "pork_weighing"
        logger.info(f"[DISPATCH] Job #{job_id} — db agent='{job.agent}', resolved='{resolved_agent}'")

        results = run_pipeline_headless(
            video_path=video_path,
            output_dir=output_dir,
            roi_coords=roi_coords,
            agent=resolved_agent,
            job_config=job_config,
            progress_callback=progress_callback,
            job_id=job_id,
        )

        # Upload verification frames to S3 for persistent storage
        if output_dir:
            try:
                from s3_manager import upload_frames_to_s3, save_s3_urls
                p2_frames = os.path.join(output_dir, "phase2_frames")
                p1_frames = os.path.join(output_dir, "phase1_event_frames")
                frames_dir_for_s3 = p2_frames if os.path.exists(p2_frames) else (
                    p1_frames if os.path.exists(p1_frames) else None
                )
                if frames_dir_for_s3:
                    url_map = upload_frames_to_s3(frames_dir_for_s3, str(job_id))
                    if url_map:
                        save_s3_urls(output_dir, url_map)
                        logger.info(f"S3: uploaded {len(url_map)} frames for job {job_id}")
            except Exception as s3_err:
                logger.warning(f"S3 frame upload failed (non-fatal): {s3_err}")

        # Persist Phase 1 detections separately so the dashboard can show them
        phase1_dets_raw = results.get("phase1_detections", [])
        if phase1_dets_raw and output_dir:
            import json as _json
            p1_path = os.path.join(output_dir, "phase1_detections.json")
            with open(p1_path, "w", encoding="utf-8") as _f:
                _json.dump(phase1_dets_raw, _f, indent=2, default=str)

        # Save detection results to DB
        from timestamp_ocr import compute_real_timestamps_for_results
        detections = results.get("phase2_detections") or results.get("phase1_detections", [])
        # Use the correct video-seconds field per agent for real_timestamp computation
        _ts_field = {
            "serve_time":          "video_seated_time",
            "bowl_completion_rate": "video_timestamp",
        }.get(job.agent, "start_time")
        detections = compute_real_timestamps_for_results(detections, recording_hour, time_field=_ts_field)
        for idx, det in enumerate(detections, 1):
            if job.agent == "plating_time":
                # For Plating Time, map video times to the DB start/end float columns
                start_val = det.get("video_start_time", 0)
                end_val = det.get("video_end_time", 0)
                scale_val = det.get("bowl_id")
                phase1_val = None
                verified_val = det.get("plating_time_seconds")
                unit_val = "sec"
                # Store the OCR timestamps in description
                desc = f"[{det.get('start_time')} - {det.get('end_time')}] {det.get('description', '')}"
            elif job.agent == "serve_time":
                start_val = det.get("video_seated_time", 0)
                end_val = det.get("video_serving_time", 0)
                scale_val = det.get("bowl_completion_status") or det.get("customer_id")
                phase1_val = None
                verified_val = det.get("service_time_seconds")
                unit_val = "sec"
                desc = f"[{det.get('seated_time')} → {det.get('serving_time')}] {det.get('bowl_completion_status', '')} | {det.get('remaining_contents', '')} | {det.get('description', '')}"
            elif job.agent == "bowl_completion_rate":
                start_val = det.get("video_timestamp", 0)
                end_val = det.get("video_timestamp", 0)
                scale_val = det.get("is_completed")
                phase1_val = None
                # Store completion as numeric so the result filter doesn't drop it
                verified_val = 1.0 if det.get("is_completed") == "COMPLETED" else 0.0
                unit_val = None
                desc = det.get("remarks", "")
            elif job.agent == "noodle_rotation":
                start_val = det.get("start_time", 0)
                end_val = det.get("end_time", 0)
                
                cw_strokes = det.get("rotation_strokes_cw") or 0
                ccw_strokes = det.get("rotation_strokes_ccw") or 0
                total_strokes = cw_strokes + ccw_strokes
                
                scale_val = total_strokes
                phase1_val = None
                verified_val = total_strokes
                unit_val = "rotations"
                desc = det.get("description", "")
            else:
                start_val = det.get("start_time", 0)
                end_val = det.get("end_time", 0)
                scale_val = det.get("scale")
                phase1_val = det.get("scale_reading")
                verified_val = det.get("verified_reading", det.get("scale_reading"))
                unit_val = det.get("unit")
                desc = det.get("description", "")

            result_record = Result(
                job_id=job_id,
                event_id=det.get("event_id") or idx,
                start_time=start_val,
                end_time=end_val,
                scale=scale_val,
                phase1_reading=phase1_val,
                verified_reading=verified_val,
                unit=unit_val,
                reading_state=det.get("reading_state"),
                confidence=det.get("confidence", 0),
                description=desc,
                reading_correction=det.get("reading_correction"),
                real_timestamp=det.get("real_timestamp"),
            )
            session.add(result_record)

        # Save token usage to DB
        token_usage = results.get("token_usage", {})
        for batch_info in token_usage.get("phase1_batches", []):
            token_record = TokenUsage(
                job_id=job_id,
                phase="phase1",
                batch_or_clip_id=batch_info.get("batch_id"),
                prompt_tokens=batch_info.get("prompt_tokens", 0),
                completion_tokens=batch_info.get("completion_tokens", 0),
                total_tokens=batch_info.get("total_tokens", 0),
            )
            session.add(token_record)

        for clip_info in token_usage.get("phase2_clips", []):
            token_record = TokenUsage(
                job_id=job_id,
                phase="phase2",
                batch_or_clip_id=clip_info.get("clip_index"),
                prompt_tokens=clip_info.get("prompt_tokens", 0),
                completion_tokens=clip_info.get("completion_tokens", 0),
                total_tokens=clip_info.get("total_tokens", 0),
            )
            session.add(token_record)

        # Update job as completed
        _update_job(session, job_id,
                    status="completed",
                    phase1_count=results.get("phase1_count", 0),
                    phase2_count=results.get("phase2_count", 0),
                    total_tokens=results.get("total_tokens", 0),
                    progress_message="Analysis complete!")

        session.commit()
        logger.info(f"✅ Job {job_id} completed successfully.")

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)
        try:
            _update_job(session, job_id,
                        status="failed",
                        error_message=str(e)[:1000],
                        progress_message=f"Failed: {str(e)[:200]}")
            session.commit()
        except Exception:
            session.rollback()

    finally:
        session.close()


def worker_loop(poll_interval: float = 30.0):
    """
    Main worker loop. Uses the in-memory job cache to find queued jobs so
    no Firestore reads are needed for polling. Falls back to a direct DB
    query only when the cache is not yet initialised (SQLite mode or startup
    race).
    Runs until shutdown flag is set.
    """
    logger.info("🔧 Background worker started.")
    import job_cache

    while not _shutdown_flag.is_set():
        session = None
        try:
            # ── Find next queued job ──────────────────────────────────────────
            queued_job = None

            if job_cache.is_ready():
                # Fast path: serve from in-memory cache (zero Firestore reads)
                candidates = sorted(
                    [j for j in (job_cache.get_all() or []) if j.get("status") == "queued"],
                    key=lambda j: j.get("created_at") or "",
                )
                if candidates:
                    queued_job = candidates[0]
            else:
                # Fallback: direct DB query (SQLite mode or cache not yet ready)
                session = SessionLocal()
                db_job = (session.query(Job)
                          .filter(Job.status == "queued")
                          .order_by(Job.created_at.asc())
                          .first())
                if db_job:
                    queued_job = {"id": db_job.id, "video_name": db_job.video_name}
                session.close()
                session = None

            if queued_job:
                job_id = queued_job["id"]
                video_name = queued_job.get("video_name", "")
                logger.info(f"🎬 Processing job #{job_id}: {video_name}")
                process_single_job(job_id)
                logger.info(f"✅ Worker loop: job #{job_id} finished, polling for next.")
            else:
                # Wait for a queued-job signal or the fallback timeout
                job_cache.wait_for_queued(timeout=poll_interval)

        except (KeyboardInterrupt, SystemExit):
            # Propagate clean shutdown signals — exit the loop
            logger.info("Worker loop: received shutdown signal, stopping.")
            break

        except BaseException as e:
            # Catch ALL exceptions (including non-Exception BaseException subclasses)
            # so the thread never dies silently between jobs.
            logger.error(f"Worker loop error ({type(e).__name__}): {e}", exc_info=True)
            _shutdown_flag.wait(timeout=poll_interval)

        finally:
            # Always ensure the polling session is closed, even on unexpected errors.
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

    logger.info("🛑 Background worker stopped.")


def start_worker_thread() -> threading.Thread:
    """Start the background worker in a daemon thread. Returns the thread.

    Clears _shutdown_flag before starting so that a previously stopped worker
    (e.g. from a shutdown/restart cycle) doesn't immediately exit on the first
    loop iteration.
    """
    _shutdown_flag.clear()  # reset so the new thread doesn't exit immediately
    thread = threading.Thread(target=worker_loop, daemon=True, name="bg-worker")
    thread.start()
    return thread


def stop_worker():
    """Signal the worker to stop and flush Langfuse."""
    _shutdown_flag.set()
    try:
        from langfuse_manager import flush as langfuse_flush
        langfuse_flush()
    except Exception:
        pass


# =============================================================================
# Standalone worker entry point
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("worker.log", encoding="utf-8"),
        ]
    )

    init_db()

    def signal_handler(*_):
        logger.info("Received shutdown signal...")
        stop_worker()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("🥩 Pork Weighing Analysis — Background Worker")
    print("=" * 60)
    print("Polling for jobs... Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        worker_loop(poll_interval=3.0)
    except KeyboardInterrupt:
        stop_worker()
        print("\nWorker stopped.")
