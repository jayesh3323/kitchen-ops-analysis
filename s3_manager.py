"""
AWS S3 Manager for verification frame storage.
Uploads Phase 2 verification frames to S3 for persistent access
(HF Spaces filesystem is ephemeral and resets on restart).

Requires env vars: USE_AWS_S3=true, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                   AWS_S3_BUCKET_NAME, AWS_S3_REGION (default: us-east-1)
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore[assignment]
    logger.warning("boto3 not installed — AWS S3 upload will be skipped. Run: pip install boto3")


def _get_s3_client():
    """Create an authenticated S3 client. Returns None if credentials are absent."""
    if not _BOTO3_AVAILABLE:
        return None
    try:
        import config as app_config
        if not app_config.AWS_ACCESS_KEY_ID or not app_config.AWS_SECRET_ACCESS_KEY:
            return None
        return boto3.client(
            "s3",
            aws_access_key_id=app_config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=app_config.AWS_SECRET_ACCESS_KEY,
            region_name=app_config.AWS_S3_REGION or "us-east-1",
        )
    except Exception as e:
        logger.warning(f"Failed to create S3 client: {e}")
        return None


def upload_frames_to_s3(frames_dir: str, job_id: str) -> Dict[str, str]:
    """
    Upload all PNG/JPEG frames in frames_dir (recursively) to S3.

    S3 key format: verification_frames/job_{job_id}/{relative_path}
    Returns a dict mapping local absolute path -> public S3 URL.
    Returns empty dict if S3 is not configured or upload fails.
    """
    try:
        import config as app_config
        if not app_config.USE_AWS_S3:
            return {}
    except Exception:
        return {}

    s3_client = _get_s3_client()
    if not s3_client:
        logger.warning("S3 not configured — skipping frame upload")
        return {}

    try:
        import config as app_config
        bucket = app_config.AWS_S3_BUCKET_NAME
        region = app_config.AWS_S3_REGION or "us-east-1"
    except Exception:
        return {}

    if not bucket:
        logger.warning("AWS_S3_BUCKET_NAME not set — skipping frame upload")
        return {}

    url_map: Dict[str, str] = {}
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        return {}

    for fpath in sorted(frames_path.rglob("*")):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        rel = fpath.relative_to(frames_path)
        s3_key = f"verification_frames/job_{job_id}/{rel.as_posix()}"
        content_type = "image/png" if fpath.suffix.lower() == ".png" else "image/jpeg"

        try:
            s3_client.upload_file(
                str(fpath),
                bucket,
                s3_key,
                ExtraArgs={"ContentType": content_type},
            )
            # Build public URL (works for public-read ACL or presigned URLs)
            url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
            url_map[str(fpath)] = url
            logger.info(f"S3 upload OK: {s3_key}")
        except Exception as e:
            logger.warning(f"Failed to upload {fpath.name} to S3: {e}")

    logger.info(f"Uploaded {len(url_map)} frames to S3 for job {job_id}")
    return url_map


def generate_presigned_url(s3_key: str, expiry_seconds: int = 3600) -> Optional[str]:
    """Generate a presigned URL for a private S3 object."""
    s3_client = _get_s3_client()
    if not s3_client:
        return None
    try:
        import config as app_config
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": app_config.AWS_S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=expiry_seconds,
        )
        return url
    except Exception as e:
        logger.warning(f"Failed to generate presigned URL for {s3_key}: {e}")
        return None


def save_s3_urls(output_dir: str, url_map: Dict[str, str]) -> None:
    """Persist S3 URL map to {output_dir}/s3_frame_urls.json for later retrieval."""
    if not url_map:
        return
    path = os.path.join(output_dir, "s3_frame_urls.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(url_map, f, indent=2)
        logger.info(f"Saved {len(url_map)} S3 URLs to {path}")
    except Exception as e:
        logger.warning(f"Failed to save S3 URL map: {e}")


def load_s3_urls(output_dir: str) -> Dict[str, str]:
    """Load the S3 URL map from {output_dir}/s3_frame_urls.json."""
    path = os.path.join(output_dir, "s3_frame_urls.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load S3 URL map: {e}")
        return {}


def list_s3_frames_as_api_items(url_map: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Convert url_map (local_path -> s3_url) into the list format expected by
    the /api/jobs/{id}/frames endpoint: [{filename, url, mime}]
    """
    items = []
    for local_path, url in sorted(url_map.items(), key=lambda x: x[0]):
        fname = os.path.basename(local_path)
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "png"
        mime = "image/png" if ext == "png" else "image/jpeg"
        items.append({"filename": fname, "url": url, "mime": mime})
    return items
