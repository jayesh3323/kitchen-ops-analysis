import sys
from pathlib import Path
from huggingface_hub import HfApi

SPACE_ID = "Jayesh-Ah/kitchen-ops-dashboard"
WEBAPP_DIR = Path(__file__).parent
api = HfApi()

FILES_TO_UPLOAD = []
core_files = [
    "main.py", "config.py", "database.py", "firebase_db.py", "worker.py",
    "pipeline_adapter.py", "langfuse_manager.py", "timestamp_ocr.py",
    "upload_prompts.py", "requirements.txt", "Dockerfile", ".dockerignore"
]
for f in core_files:
    if (WEBAPP_DIR / f).exists(): FILES_TO_UPLOAD.append((str(WEBAPP_DIR / f), f))

for f in (WEBAPP_DIR / "static").glob("*"):
    if f.is_file(): FILES_TO_UPLOAD.append((str(f), f"static/{f.name}"))

for f in (WEBAPP_DIR / "templates").glob("*"):
    if f.is_file(): FILES_TO_UPLOAD.append((str(f), f"templates/{f.name}"))

agents_dir = WEBAPP_DIR / "agents"
if agents_dir.exists():
    for f in agents_dir.glob("*.py"):
        if not f.name.startswith("__"): FILES_TO_UPLOAD.append((str(f), f"agents/{f.name}"))

api.upload_folder(
    folder_path=str(WEBAPP_DIR),
    repo_id=SPACE_ID,
    repo_type="space",
    allow_patterns=[r for _, r in FILES_TO_UPLOAD],
    commit_message="Feature: Firebase Auth & Direct Browser-to-Storage Video Uploads"
)
print("Push complete!")
