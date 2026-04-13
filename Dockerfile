# ============================================================================
# Kitchen Ops Dashboard — Hugging Face Spaces Docker Image
# ============================================================================
# Deploy as a Docker Space on HF. Port 7860 is the default for HF Spaces.
#
# REPO STRUCTURE (in the HF Space repo):
#   Dockerfile
#   requirements.txt
#   main.py, config.py, database.py, firebase_db.py, worker.py, ...
#   pipeline_adapter.py, langfuse_manager.py, timestamp_ocr.py
#   agents/                     (pipeline scripts)
#   static/                     (app.js, style.css)
#   templates/                  (index.html)
#
# Required HF Secrets:
#   FIREBASE_SERVICE_ACCOUNT_JSON  (raw JSON from Firebase service account)
#   OPENAI_API_KEY
#   GOOGLE_API_KEY
#   LANGFUSE_PUBLIC_KEY
#   LANGFUSE_SECRET_KEY
# ============================================================================

FROM python:3.11-slim

# Install system dependencies (ffmpeg for video processing, OpenCV deps)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy all application code
COPY . .

# Create directories the app expects
RUN mkdir -p /app/uploads /app/results /tmp/video_processing

ENV PORT=7860

# Environment defaults for HF Spaces (overridden by HF Secrets)
ENV USE_FIREBASE=true
ENV DATABASE_URL=sqlite:///./jobs.db
ENV UPLOAD_DIR=/app/uploads
ENV RESULTS_DIR=/app/results
ENV LANGFUSE_HOST=https://cloud.langfuse.com

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
