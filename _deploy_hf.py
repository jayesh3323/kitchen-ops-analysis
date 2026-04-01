"""
Deploy Kitchen Ops Dashboard to Hugging Face Spaces.
Creates the Space (if needed) and uploads all required files.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

SPACE_ID = "Jayesh-Ah/kitchen-ops-analysis"
WEBAPP_DIR = Path(__file__).parent

api = HfApi()

# Step 1: Define all files to upload
FILES_TO_UPLOAD = []

core_files = [
    "main.py",
    "config.py",
    "database.py",
    "firebase_db.py",
    "worker.py",
    "pipeline_adapter.py",
    "langfuse_manager.py",
    "timestamp_ocr.py",
    "job_cache.py",
    "auto_roi.py",
    "upload_prompts.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
]

for f in core_files:
    fpath = WEBAPP_DIR / f
    if fpath.exists():
        FILES_TO_UPLOAD.append((str(fpath), f))

readme_path = WEBAPP_DIR / "README_HF.md"
# README is handled separately via upload_file — do NOT add to FILES_TO_UPLOAD
# to avoid upload_folder accidentally uploading the local README.md (no HF frontmatter)

for f in (WEBAPP_DIR / "static").glob("*"):
    if f.is_file():
        FILES_TO_UPLOAD.append((str(f), f"static/{f.name}"))

for f in (WEBAPP_DIR / "templates").glob("*"):
    if f.is_file():
        FILES_TO_UPLOAD.append((str(f), f"templates/{f.name}"))

agents_dir = WEBAPP_DIR / "agents"
if agents_dir.exists():
    for f in agents_dir.glob("*.py"):
        if f.name.startswith("__"):
            continue
        FILES_TO_UPLOAD.append((str(f), f"agents/{f.name}"))

print(f"Uploading {len(FILES_TO_UPLOAD)} files to {SPACE_ID} ...")

# Build a multi-file commit: app code + README in one shot (single rebuild)
from huggingface_hub import CommitOperationAdd

operations = []

# App code files
for local_path, repo_path in FILES_TO_UPLOAD:
    operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path))

# README with HF frontmatter — must be last so it's never overwritten by a stale local README.md
if readme_path.exists():
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_path)))

api.create_commit(
    repo_id=SPACE_ID,
    repo_type="space",
    operations=operations,
    commit_message="Fix slow startup: non-blocking DB init thread; guard list_jobs during Firebase init",
)

print("DONE!")
