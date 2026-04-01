"""
FastAPI server for the Pork Weighing Analysis Web App.
Provides REST API endpoints, serves the dashboard UI, and runs a background worker.
No Redis/Celery/Langfuse required — uses a pure-Python SQLite-polling worker thread.
"""
import os
import json
import shutil
import base64
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config as app_config

from database import init_db, SessionLocal, Job
from worker import start_worker_thread, stop_worker

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Pork Weighing Analysis",
    description="Web-based video analysis pipeline for pork weighing event detection",
    version="1.0.0",
)

# Static files and templates
_webapp_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_webapp_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(_webapp_dir / "templates"))

# Background worker thread reference
_worker_thread = None


def _watchdog_loop(poll_seconds: float = 10.0):
    """
    Monitors the worker thread and restarts it if it dies unexpectedly.
    Runs as its own daemon thread so it never blocks the server.
    """
    import time
    global _worker_thread
    while True:
        time.sleep(poll_seconds)
        from worker import _shutdown_flag
        if _shutdown_flag.is_set():
            # Server is shutting down intentionally — stop watching.
            break
        if _worker_thread is None or not _worker_thread.is_alive():
            logger.warning("⚠️  Worker thread died unexpectedly — restarting.")
            _worker_thread = start_worker_thread()
            logger.info("🔁 Worker thread restarted.")


@app.on_event("startup")
def startup():
    """Initialize database and start background worker on server startup.

    init_db() is intentionally run in a background thread so uvicorn can
    start accepting connections (and pass HF Spaces' health check) immediately.
    Firebase/Firestore init involves network I/O that can block for 60–120 s
    when Application Default Credentials fall back to the GCP metadata server.
    """
    global _worker_thread

    def _init_db_bg():
        try:
            init_db()
            logger.info("Database initialized.")
        except Exception as exc:
            logger.error(f"Database init failed (non-fatal): {exc}")

    threading.Thread(target=_init_db_bg, daemon=True, name="db-init").start()

    logger.info(f"Upload dir: {app_config.UPLOAD_DIR}")
    logger.info(f"Results dir: {app_config.RESULTS_DIR}")
    _worker_thread = start_worker_thread()
    logger.info("Background worker thread started.")
    # Start watchdog so the worker is automatically restarted if it ever dies.
    watchdog = threading.Thread(target=_watchdog_loop, daemon=True, name="worker-watchdog")
    watchdog.start()
    logger.info("Worker watchdog started.")


@app.on_event("shutdown")
def shutdown():
    """Stop background worker on server shutdown."""
    stop_worker()
    logger.info("Background worker stopped.")


# =============================================================================
# Pages
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "max_upload_mb": app_config.MAX_UPLOAD_SIZE_MB,
    })


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/api/upload-video")
async def upload_video(video_file: UploadFile = File(...)):
    """
    Pre-upload a video file to the server. Returns the server-side path.
    Used by the frontend to upload once, then reuse the path for both
    ROI auto-detection and job submission — avoiding double uploads.
    """
    video_name = video_file.filename or "upload.mp4"
    upload_path = os.path.join(app_config.UPLOAD_DIR, video_name)

    # Handle duplicate filenames
    counter = 1
    base_name = Path(upload_path).stem
    extension = Path(upload_path).suffix
    while os.path.exists(upload_path):
        upload_path = os.path.join(app_config.UPLOAD_DIR, f"{base_name}_{counter}{extension}")
        counter += 1

    file_size = 0
    try:
        with open(upload_path, "wb") as f:
            while chunk := await video_file.read(8192):
                file_size += len(chunk)
                if file_size > app_config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    os.remove(upload_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {app_config.MAX_UPLOAD_SIZE_MB}MB limit"
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    logger.info(f"Pre-uploaded video: {upload_path} ({file_size / 1024 / 1024:.1f} MB)")
    return JSONResponse({
        "path": upload_path,
        "filename": Path(upload_path).name,
        "size_mb": round(file_size / 1024 / 1024, 1),
    })


@app.post("/api/jobs")
async def create_job(
    video_file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    # ROI region
    roi_x1: Optional[int] = Form(None),
    roi_y1: Optional[int] = Form(None),
    roi_x2: Optional[int] = Form(None),
    roi_y2: Optional[int] = Form(None),
    # Timestamp region (for OCR crop)
    ts_x1: Optional[int] = Form(None),
    ts_y1: Optional[int] = Form(None),
    ts_x2: Optional[int] = Form(None),
    ts_y2: Optional[int] = Form(None),
    # Config
    enable_phase2: bool = Form(True),
    fps: Optional[float] = Form(None),
    recording_hour: Optional[int] = Form(None),
    recording_date: Optional[str] = Form(None),
    agent: Optional[str] = Form(None),
    advanced_config: Optional[str] = Form(None),
    original_filename: Optional[str] = Form(None),
):
    """Create a new video analysis job. The background worker will pick it up automatically."""
    db = SessionLocal()
    try:
        final_video_path = None
        video_name = None

        if video_file and video_file.filename:
            file_size = 0
            video_name = video_file.filename
            upload_path = os.path.join(app_config.UPLOAD_DIR, video_name)

            # Handle duplicate filenames
            counter = 1
            base_name = Path(video_name).stem
            extension = Path(video_name).suffix
            while os.path.exists(upload_path):
                video_name = f"{base_name}_{counter}{extension}"
                upload_path = os.path.join(app_config.UPLOAD_DIR, video_name)
                counter += 1

            with open(upload_path, "wb") as f:
                while chunk := await video_file.read(8192):
                    file_size += len(chunk)
                    if file_size > app_config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                        os.remove(upload_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds {app_config.MAX_UPLOAD_SIZE_MB}MB limit"
                        )
                    f.write(chunk)

            final_video_path = upload_path
            logger.info(f"Uploaded video: {upload_path} ({file_size / 1024 / 1024:.1f} MB)")

        elif video_path:
            is_gs = video_path.startswith("gs://")
            if not is_gs and not os.path.exists(video_path):
                raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
            
            final_video_path = video_path
            video_name = original_filename or (video_path.split("/")[-1] if is_gs else Path(video_path).name)
        else:
            raise HTTPException(status_code=400, detail="Provide either a video file or a file path.")

        # Build ROI coords JSON
        roi_coords = None
        if all(v is not None for v in [roi_x1, roi_y1, roi_x2, roi_y2]):
            roi_coords = json.dumps([roi_x1, roi_y1, roi_x2, roi_y2])

        # Build timestamp region coords JSON
        timestamp_region_coords = None
        if all(v is not None for v in [ts_x1, ts_y1, ts_x2, ts_y2]):
            timestamp_region_coords = json.dumps([ts_x1, ts_y1, ts_x2, ts_y2])

        job_config = {
            "enable_phase2": enable_phase2,
            "fps": fps or app_config.FPS,
        }
        
        if advanced_config:
            try:
                adv = json.loads(advanced_config)
                job_config.update(adv)
            except Exception as e:
                logger.warning(f"Failed to parse advanced_config: {e}")

        job = Job(
            video_name=video_name,
            video_path=final_video_path,
            status="queued",
            roi_coords=roi_coords,
            timestamp_region_coords=timestamp_region_coords,
            config_json=json.dumps(job_config),
            progress_message="Queued for processing...",
            recording_hour=recording_hour,
            recording_date=recording_date,
            agent=agent,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        logger.info(f"Created job #{job.id} for video '{video_name}'")

        return JSONResponse(status_code=201, content={
            "id": job.id,
            "status": "queued",
            "message": f"Job #{job.id} created. Processing will begin shortly.",
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


def _is_quota_error(exc: Exception) -> bool:
    """Return True if the exception is a Firestore/gRPC quota-exceeded error."""
    name = type(exc).__name__
    msg = str(exc)
    return (
        "ResourceExhausted" in name
        or "Quota exceeded" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or getattr(exc, "grpc_status_code", None) is not None and "RESOURCE_EXHAUSTED" in str(getattr(exc, "grpc_status_code", ""))
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs ordered by creation time (newest first). Served from cache."""
    import job_cache
    cached = job_cache.get_all()
    if cached is not None:
        return cached
    # Cache not yet populated (Firebase still initializing) → return empty list
    if app_config.USE_FIREBASE:
        return []
    # Fallback for SQLite mode
    db = SessionLocal()
    try:
        jobs = db.query(Job).order_by(Job.created_at.desc()).limit(100).all()
        return [job.to_dict() for job in jobs]
    except Exception as e:
        if _is_quota_error(e):
            logger.warning(f"Firestore quota exceeded on list_jobs: {e}")
            raise HTTPException(status_code=429, detail="Quota exceeded — retry shortly.")
        raise
    finally:
        db.close()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a single job with its results. Metadata from cache; results cached after first load."""
    import job_cache

    # ── Job metadata ─────────────────────────────────────────────────────────
    job_data = job_cache.get(job_id)
    agent_name = ""
    config_json_str = None

    if job_data is not None:
        # Serve metadata from cache (no Firestore read)
        job_data = dict(job_data)  # shallow copy so we can add keys
        agent_name = job_data.get("agent") or ""
        config_json_str = job_data.get("config_json")
    else:
        # Fallback: read from Firestore (SQLite mode or cache miss)
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            job_data = job.to_dict()
            agent_name = job_data.get("agent") or ""
            config_json_str = job_data.get("config_json")
        except HTTPException:
            raise
        except Exception as e:
            if _is_quota_error(e):
                raise HTTPException(status_code=429, detail="Quota exceeded — retry shortly.")
            raise
        finally:
            db.close()

    if job_data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # ── Results ──────────────────────────────────────────────────────────────
    def _keep_result(r: dict) -> bool:
        if agent_name == "bowl_completion_rate":
            return True
        if agent_name == "noodle_rotation":
            threshold = 0.8  
        elif agent_name == "pork_weighing_compliance":
            threshold = 0.1
        else:
            threshold = 0.80
        if (r.get("confidence") or 0) < threshold:
            return False
        if r.get("verified_reading") is None:
            return False
        return True

    job_status = job_data.get("status", "")
    raw_results = None

    # Use cached results for completed jobs; always re-fetch for in-progress jobs
    if job_status == "completed":
        raw_results = job_cache.get_results(job_id)

    if raw_results is None:
        # Read from Firestore and cache
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                raw_results = [res.to_dict() for res in job.results]
                token_usages = list(job.token_usages)
                total_p1 = sum(t.total_tokens for t in token_usages if t.phase == "phase1")
                total_p2 = sum(t.total_tokens for t in token_usages if t.phase == "phase2")
                job_data["token_summary"] = {
                    "phase1_tokens": total_p1,
                    "phase2_tokens": total_p2,
                    "total_tokens": total_p1 + total_p2,
                }
                if job_status == "completed":
                    job_cache.set_results(job_id, raw_results)
        except Exception as e:
            if _is_quota_error(e):
                raise HTTPException(status_code=429, detail="Quota exceeded — retry shortly.")
            raise
        finally:
            db.close()

    if raw_results is None:
        raw_results = []

    if "token_summary" not in job_data:
        job_data["token_summary"] = {"phase1_tokens": 0, "phase2_tokens": 0, "total_tokens": 0}

    job_data["results"] = [r for r in raw_results if _keep_result(r)]

    # ── Config ───────────────────────────────────────────────────────────────
    config = {}
    try:
        config = json.loads(config_json_str) if config_json_str else {}
    except (json.JSONDecodeError, TypeError):
        pass
    job_data["config"] = config
    job_data["phase2_enabled"] = config.get("enable_phase2", True)

    return job_data


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and ALL associated data: DB records, uploaded video, output directory."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        deleted_paths = []

        # 1. Delete uploaded video (only if it lives inside UPLOAD_DIR — not user-provided paths)
        if job.video_path and job.video_path.startswith(app_config.UPLOAD_DIR):
            if os.path.exists(job.video_path):
                os.remove(job.video_path)
                deleted_paths.append(f"video:{job.video_path}")

        # 2. Delete output directory (results, clips, frames, CSVs)
        if job.output_dir and os.path.exists(job.output_dir):
            shutil.rmtree(job.output_dir, ignore_errors=True)
            deleted_paths.append(f"output_dir:{job.output_dir}")

        # 3. Delete DB record — cascade="all, delete-orphan" removes Result + TokenUsage rows
        db.delete(job)
        db.commit()

        logger.info(f"Deleted job #{job_id}: {', '.join(deleted_paths) or 'no files'}")
        return {"message": f"Job #{job_id} deleted.", "deleted": deleted_paths}
    finally:
        db.close()


@app.get("/api/jobs/{job_id}/frames")
async def get_verification_frames(job_id: str):
    """
    List Phase 2 verification frame images for a completed job.
    Returns a list of base64-encoded images.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if not job.output_dir:
            return {"frames": []}

        frames = []

        def _walk_dir_into_frames(scan_dir: str, label_prefix: str = "") -> None:
            for root, _dirs, files in os.walk(scan_dir):
                for fname in sorted(files):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        fpath = os.path.join(root, fname)
                        with open(fpath, "rb") as f:
                            data = base64.b64encode(f.read()).decode("utf-8")
                        ext = fname.rsplit(".", 1)[-1].lower()
                        mime = "image/png" if ext == "png" else "image/jpeg"
                        rel = os.path.relpath(fpath, scan_dir).replace("\\", "/")
                        filename = (label_prefix + rel) if label_prefix else rel
                        frames.append({"filename": filename, "data": data, "mime": mime, "url": None})

        # Always include CLAHE preview frames if present (pork_weighing agent).
        # These are saved regardless of whether any events were detected.
        clahe_dir = os.path.join(job.output_dir, "clahe_preview_frames")
        if os.path.exists(clahe_dir):
            _walk_dir_into_frames(clahe_dir, label_prefix="clahe_preview/")

        # Detection frames — priority: phase2_frames → phase1_event_frames → event_frames
        #   1. phase2_frames/  — subdirs event_N/ or detection_NNN/ (pork, noodle, bowl)
        #   2. phase1_event_frames/ — subdirs event_N/ (pork, noodle)
        #   3. event_frames/   — flat files phase{1|2}_event{N}_... (plating, serve_time)
        p2_dir = os.path.join(job.output_dir, "phase2_frames")
        p1_dir = os.path.join(job.output_dir, "phase1_event_frames")
        ev_dir = os.path.join(job.output_dir, "event_frames")

        if os.path.exists(p2_dir):
            _walk_dir_into_frames(p2_dir)
        elif os.path.exists(p1_dir):
            _walk_dir_into_frames(p1_dir)
        elif os.path.exists(ev_dir):
            _walk_dir_into_frames(ev_dir)

        # If no local frames found, try S3 URLs (HF Spaces filesystem is ephemeral)
        if not frames:
            try:
                from s3_manager import load_s3_urls, list_s3_frames_as_api_items
                url_map = load_s3_urls(job.output_dir)
                if url_map:
                    frames = [
                        {**item, "data": None}
                        for item in list_s3_frames_as_api_items(url_map)
                    ]
            except Exception as s3_err:
                logger.warning(f"Failed to load S3 frame URLs for job {job_id}: {s3_err}")

        return {"frames": frames}
    finally:
        db.close()


@app.post("/api/extract-frame")
async def extract_frame(
    video_file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    frame_index: int = Form(30),  # Changed default to 30 to match ROI step logic
):
    """
    Extract a single frame from a video for ROI / timestamp region selection.
    Returns the frame as a base64-encoded JPEG.
    """
    from pipeline_adapter import extract_frame_for_roi

    temp_path = None
    try:
        if video_file and video_file.filename:
            # Use chunked writing to handle large files without memory exhaustion
            temp_path = os.path.join(app_config.UPLOAD_DIR, f"_temp_{video_file.filename}")
            with open(temp_path, "wb") as f:
                while True:
                    chunk = await video_file.read(1024 * 1024 * 5)  # 5MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
            target_path = temp_path
        elif video_path:
            if not os.path.exists(video_path):
                raise HTTPException(status_code=400, detail="Video file not found")
            target_path = video_path
        else:
            raise HTTPException(status_code=400, detail="Provide a video file or path")

        frame_bytes = extract_frame_for_roi(target_path, frame_index)
        if frame_bytes is None:
            logger.error(f"Failed to extract frame at index {frame_index} from {target_path}")
            raise HTTPException(status_code=500, detail="Failed to extract frame from video. File may be corrupted or unreadable.")

        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
        return {"frame": frame_b64, "format": "jpeg"}

    except Exception as e:
        logger.error(f"Error in extract-frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_path}: {e}")


@app.post("/api/extract-timestamp")
async def extract_timestamp_endpoint(request: Request):
    """
    Run GPT OCR on a base64 frame (or cropped region) to extract recording date and hour.
    Body: { "frame_b64": "...", "region": [x1,y1,x2,y2] (optional) }
    """
    from timestamp_ocr import extract_timestamp_from_frame
    from openai import OpenAI
    import cv2
    import numpy as np

    try:
        body = await request.json()
        frame_b64 = body.get("frame_b64")
        if not frame_b64:
            raise HTTPException(status_code=400, detail="Missing frame_b64")

        # Optionally crop to the selected region before OCR
        region = body.get("region")
        if region and len(region) == 4:
            try:
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",", 1)[1]
                nparr = np.frombuffer(base64.b64decode(frame_b64), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Could not decode base64 image")

                x1, y1, x2, y2 = [int(v) for v in region]
                h, w = img.shape[:2]

                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Invalid crop region: {x1},{y1} -> {x2},{y2}")

                cropped = img[y1:y2, x1:x2]

                # The frame shown to the user was already rotated by `rot` degrees.
                # The user drew coordinates in that rotated space.
                # To make the timestamp text horizontal for OCR, apply the inverse rotation.
                #
                # Forward rotation → inverse rotation needed on crop:
                #   0°   → no-op
                #   90°  CW  → 90° CCW  (ROTATE_90_COUNTERCLOCKWISE)
                #   180°     → 180°     (ROTATE_180)
                #   270° CW  → 90°  CW  (ROTATE_90_CLOCKWISE)  [same as 90° CCW of 270°]
                rot = body.get("rotation_angle")
                if rot is None:
                    rot = app_config.ROTATION_ANGLE
                rot = int(rot)
                if rot == 270:
                    cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
                    logger.info("Inverse-rotated crop 90° CW (frame was 270° CCW)")
                elif rot == 90:
                    cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    logger.info("Inverse-rotated crop 90° CCW (frame was 90° CW)")
                elif rot == 180:
                    cropped = cv2.rotate(cropped, cv2.ROTATE_180)
                    logger.info("Inverse-rotated crop 180° (frame was 180°)")
                # rot == 0: text already horizontal, no rotation needed

                # Save debug crop
                debug_path = os.path.join(app_config.UPLOAD_DIR, "debug_timestamp_crop.jpg")
                cv2.imwrite(debug_path, cropped)
                logger.info(f"Saved debug timestamp crop to {debug_path}")

                _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                logger.info(f"Cropped frame to region: {region}. New size: {cropped.shape}")
            except Exception as e:
                logger.error(f"Failed to crop image: {e}")
                # Fallback to full frame if crop fails, or raise error?
                # Raising error is better so user knows something is wrong
                raise HTTPException(status_code=400, detail=f"Failed to process image crop: {str(e)}")

        ocr_client = OpenAI(api_key=app_config.OPENAI_API_KEY)
        recording_date, recording_hour, raw_timestamp = extract_timestamp_from_frame(
            frame_b64, ocr_client, model=app_config.PHASE1_MODEL_NAME
        )

        logger.info(f"OCR Result - Date: {recording_date}, Hour: {recording_hour}")

        return {
            "recording_date": recording_date,
            "recording_hour": recording_hour,
            "raw_timestamp": raw_timestamp,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Timestamp OCR endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto-detect-timestamp")
async def auto_detect_timestamp_endpoint(
    video_file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    rotation_angle: Optional[int] = Form(None),
    agent: Optional[str] = Form(None),
):
    """
    Auto-detect and OCR the timestamp from the bottom-left corner of an early video frame.
    Returns JSON: { recording_date, recording_hour, raw_timestamp, annotated_frame (base64 JPEG) }
    """
    import cv2
    import numpy as np
    from openai import OpenAI
    from timestamp_ocr import extract_timestamp_from_frame

    temp_path = None
    try:
        if video_file and video_file.filename:
            temp_path = os.path.join(app_config.UPLOAD_DIR, f"_ts_temp_{video_file.filename}")
            with open(temp_path, "wb") as f:
                while True:
                    chunk = await video_file.read(1024 * 1024 * 5)
                    if not chunk:
                        break
                    f.write(chunk)
            target_path = temp_path
        elif video_path:
            is_gs = video_path.startswith("gs://")
            if not is_gs and not os.path.exists(video_path):
                raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
            target_path = video_path
        else:
            raise HTTPException(status_code=400, detail="Provide a video file or path")

        # If caller didn't supply a rotation, use the agent's own default.
        agent_name = agent or "pork_weighing"
        if rotation_angle is None:
            ad = app_config.get_agent_defaults(agent_name)
            rotation_angle = int(ad.get("AGENT_ROTATION_ANGLE", app_config.ROTATION_ANGLE))
            logger.info(f"rotation_angle not supplied — using agent default: {rotation_angle}°")

        # ── Extract an early frame (5% into video) ────────────────────────────
        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_idx = max(0, min(int(total * 0.05), total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            raise HTTPException(status_code=500, detail="Could not extract frame from video")

        # ── Apply rotation ────────────────────────────────────────────────────
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # ── Crop full-width row y=0→700 — covers CCTV timestamp band
        h, w = frame.shape[:2]
        ts_x1, ts_y1 = 0, 0
        ts_x2, ts_y2 = w, min(700, h)
        ts_crop = frame[ts_y1:ts_y2, ts_x1:ts_x2]

        _, crop_buf = cv2.imencode(".jpg", ts_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop_b64 = base64.b64encode(crop_buf.tobytes()).decode("utf-8")

        logger.info(f"Auto-detecting timestamp from bottom-left region "
                    f"({ts_x1},{ts_y1})→({ts_x2},{ts_y2}) of {w}×{h} frame")

        # ── OCR via existing timestamp_ocr helper ─────────────────────────────
        ocr_client = OpenAI(api_key=app_config.OPENAI_API_KEY)
        recording_date, recording_hour, raw_timestamp = extract_timestamp_from_frame(
            crop_b64, ocr_client, model=app_config.PHASE1_MODEL_NAME
        )
        logger.info(f"Timestamp OCR: date={recording_date} hour={recording_hour} raw='{raw_timestamp}'")

        # ── Annotate full frame with the cropped region highlighted ───────────
        annotated = frame.copy()
        cv2.rectangle(annotated, (ts_x1, ts_y1), (ts_x2, ts_y2), (0, 200, 100), 2)
        ts_label = raw_timestamp[:40] if raw_timestamp else "Timestamp region"
        cv2.putText(annotated, ts_label, (ts_x1 + 4, max(ts_y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1, cv2.LINE_AA)
        _, ann_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_b64 = base64.b64encode(ann_buf.tobytes()).decode("utf-8")

        return JSONResponse({
            "recording_date": recording_date,
            "recording_hour": recording_hour,
            "raw_timestamp": raw_timestamp,
            "annotated_frame": annotated_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto timestamp detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove temp file {temp_path}: {cleanup_err}")


@app.post("/api/auto-detect-roi")
async def auto_detect_roi_endpoint(
    video_file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    rotation_angle: Optional[int] = Form(None),
    agent: Optional[str] = Form(None),
):
    """
    Auto-detect the ROI from a video using a VLM (GPT-4o-mini).
    The `agent` parameter tailors the prompt to the specific task context.
    Returns JSON: { roi, confidence, method, annotated_frame (base64 JPEG) }
    """
    from auto_roi import auto_detect_roi

    temp_path = None
    try:
        if video_file and video_file.filename:
            temp_path = os.path.join(app_config.UPLOAD_DIR, f"_roi_temp_{video_file.filename}")
            with open(temp_path, "wb") as f:
                while True:
                    chunk = await video_file.read(1024 * 1024 * 5)  # 5 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
            target_path = temp_path

        elif video_path:
            is_gs = video_path.startswith("gs://")
            if not is_gs and not os.path.exists(video_path):
                raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
            target_path = video_path

        else:
            raise HTTPException(status_code=400, detail="Provide a video file or path")

        agent_name = agent or "pork_weighing"
        kb_dir = getattr(app_config, "ROI_KB_DIR", None)

        # If caller didn't supply a rotation, use the agent's own default so that
        # ROI coordinates are always detected on the same orientation the pipeline
        # will use when processing analysis frames.
        if rotation_angle is None:
            ad = app_config.get_agent_defaults(agent_name)
            rotation_angle = int(ad.get("AGENT_ROTATION_ANGLE", app_config.ROTATION_ANGLE))
            logger.info(f"rotation_angle not supplied — using agent default: {rotation_angle}°")

        logger.info(
            f"Auto-detecting ROI for agent='{agent_name}': {target_path}, "
            f"rotation={rotation_angle}, kb_dir={kb_dir!r}"
        )
        result = auto_detect_roi(
            target_path,
            rotation_angle=rotation_angle,
            agent=agent_name,
            kb_dir=kb_dir,
        )

        return JSONResponse({
            "roi": result["roi"],
            "confidence": result["confidence"],
            "method": result["method"],
            "annotated_frame": base64.b64encode(result["annotated_frame"]).decode("utf-8")
                               if result.get("annotated_frame") else None,
            "frame_w": result.get("frame_w", 0),
            "frame_h": result.get("frame_h", 0),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto ROI detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove temp file {temp_path}: {cleanup_err}")


@app.post("/api/stop")
async def stop_server():
    """Shut down the server process cleanly."""
    import signal as _signal
    logger.info("Stop requested — sending SIGTERM to process %s", os.getpid())
    def _do_stop():
        import time
        time.sleep(0.5)  # Let the HTTP response be sent first
        os.kill(os.getpid(), _signal.SIGTERM)
    threading.Thread(target=_do_stop, daemon=True).start()
    return JSONResponse({"status": "stopping"})


# =============================================================================
# Run with: python -m uvicorn main:app --port 8000
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
