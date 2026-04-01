# Kitchen Ops — Video Analysis Web Application

## Overview

This is a **web-based AI video analysis platform** designed for kitchen operations monitoring. It provides a browser dashboard through which kitchen managers can upload CCTV/recording footage and automatically extract operational insights — such as pork weighing compliance, plating times, serve times, and bowl completion rates — using a two-phase AI pipeline powered by OpenAI GPT and Google Gemini models.

The application wraps existing standalone Python analysis scripts into a server-hosted job queue system, making them accessible without requiring any command-line knowledge.

---

## What We Are Building

### Core Concept

The webapp serves as a **headless orchestration layer** over multiple video analysis "agents". Each agent targets a specific kitchen KPI:

| Agent | Purpose |
|---|---|
| 🥩 **Pork Weighing Compliance** | Detects weighing events on the scale, extracts scale readings via OCR, and verifies pork weight compliance standards. |
| 🍜 **Plating Time** | Measures the time taken to complete plating of a ramen bowl from start to finish. |
| ⏱️ **Average Serve Time** | Measures the time between a customer being seated and receiving their order. |
| 🥣 **Bowl Completion Rate** | Tracks bowl completion events and rates from kitchen video. |
| 🍜 **Noodle Rotation Compliance** | Monitors noodle rotation compliance events in kitchen video. |

---

## Architecture

### Backend

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) served via [Uvicorn](https://www.uvicorn.org/)
- **Task Queue**: SQLite-backed job polling (no Redis/Celery required) — a background `worker.py` thread polls the `jobs` table for queued work and processes jobs sequentially.
- **Database**: SQLite via SQLAlchemy ORM (`jobs.db`), storing:
  - `Job` records — video path, selected agent, ROI/timestamp coordinates, config snapshot, status, and timestamps.
  - `Result` records — per-event detection results (start/end time, scale reading, confidence, real wall-clock timestamp).
  - `TokenUsage` records — LLM token consumption per phase per job.
- **Configuration**: All settings loaded from `.env` (with sensible defaults in `config.py`). The webapp `.env` overrides the parent pipeline `.env`.
- **Templating**: Jinja2 for serving the single-page `index.html` dashboard.

### Frontend

- **Single-Page App**: One HTML page (`templates/index.html`) enhanced with a vanilla JavaScript module (`static/app.js`) and custom CSS (`static/style.css`).
- **Charts**: [Chart.js](https://www.chartjs.org/) for result visualization.
- **Agent Tabs**: Top-level tab bar to switch between analysis agents.
- **Job Management Table**: Live-polling jobs table showing status, progress messages, and actions (view results, delete).

### Pipeline Adapter

`pipeline_adapter.py` acts as a bridge between the webapp and the existing standalone analysis scripts. It:
- Instantiates either `PorkWeighingPipeline` or `RamenPlatingPipeline` (or other agents) with a headless config.
- Skips interactive OpenCV ROI selection — uses coordinates stored in the job record instead.
- Reports progress to the job table via a callback function.

---

## Key Features

### 1. Video Upload & Path Input
- Drag-and-drop or file browser upload (up to 500 MB by default, configurable).
- For very large files: direct server-side file path input to bypass upload limits.

### 2. Guided ROI / Timestamp Region Selection (2-Step Flow)
- **Step 1 — Timestamp Region**: Draw a rectangle over the recording timestamp overlay in the video frame. The app OCRs this region (via GPT) to extract the recording date and hour, enabling wall-clock timestamps on all detected events.
  - Manual override fields available if OCR returns incorrect results.
- **Step 2 — Scale ROI**: Draw a rectangle over the weighing scale display area. Only this region is analyzed by the pipeline, improving accuracy and reducing token cost.
- Both selections use an interactive canvas drawn over an extracted video frame.

### 3. Two-Phase AI Analysis Pipeline
- **Phase 1 (Event Detection)**: Frames are extracted at a configurable FPS, batched, and sent to an OpenAI GPT model. The model identifies candidate events (e.g. a pork piece on the scale, plating start, customer seated).
- **Phase 2 (Verification)**: Short video clips around each candidate event are extracted and sent to Google Gemini for deep verification — confirming the event, correcting the reading, and assigning a confidence score.
- Phase 2 can be toggled on/off per job.

### 4. Advanced Configuration (per job)
Users can override `.env` defaults per job via the "Advanced Configuration" panel:
- Confidence threshold
- Max batch size (MB)
- Max frames per batch
- Batch overlap frames
- Clip buffer seconds
- Image quality (%)
- Image upscale factor
- Video rotation angle (0°, 90°, 180°, 270°)

### 5. Results Dashboard
- Per-job results panel showing detected events in a table.
- Statistics summary: total events detected, Phase 1 vs Phase 2 counts, total LLM tokens used.
- Real wall-clock timestamps computed from OCR-extracted recording hour.
- Phase 2 verification frames (base64-encoded images) viewable in the browser.
- CSV export of results.

### 6. Job History
- All past jobs persisted in SQLite.
- Job statuses: `queued → processing → phase1 → phase2 → completed / failed`.
- Jobs can be deleted (also cleans up output files).
- Live auto-refresh polling while a job is processing.

---

## Technology Stack

| Category | Library / Tool | Version |
|---|---|---|
| Web framework | FastAPI | 0.115.6 |
| ASGI server | Uvicorn | 0.34.0 |
| Templating | Jinja2 | 3.1.5 |
| ORM / Database | SQLAlchemy + SQLite | 2.0.36 |
| LLM — Phase 1 | OpenAI GPT (gpt-4o-mini / gpt-5-mini) | `openai >= 1.0.0` |
| LLM — Phase 2 | Google Gemini (gemini-2.5-pro) | `google-genai >= 1.0.0` |
| Video processing | OpenCV (headless) | `opencv-python-headless >= 4.8.0` |
| Image processing | Pillow | `>= 10.0.0` |
| Data manipulation | Pandas, NumPy | `>= 2.0.0 / 1.24.0` |
| Frontend charting | Chart.js | 4.4.0 (CDN) |
| Environment config | python-dotenv | 1.0.1 |

---

## File Structure

```
webapp/
├── main.py                  # FastAPI app, REST API endpoints
├── worker.py                # Background worker thread (SQLite polling queue)
├── pipeline_adapter.py      # Headless wrapper for analysis pipelines
├── database.py              # SQLAlchemy models: Job, Result, TokenUsage
├── config.py                # Centralized config from .env
├── timestamp_ocr.py         # GPT-based OCR for recording timestamp extraction
├── _migrate.py              # Database migration utility
├── .env                     # Local environment variables (API keys, config)
├── .env.example             # Template for .env
├── requirements.txt         # Python dependencies
├── jobs.db                  # SQLite database (auto-created)
├── agents/                  # Analysis agent modules
│   ├── avg_serve_time.py
│   ├── plating_time.py
│   ├── pork_weighing_compliance.py
│   ├── bowl_completion_rate.py
│   └── noodle_rotation_compliance.py
├── templates/
│   └── index.html           # Single-page dashboard HTML
├── static/
│   ├── app.js               # Frontend JavaScript (job management, canvas ROI, polling)
│   └── style.css            # Dashboard styling
├── uploads/                 # Uploaded video files (auto-created)
└── results/                 # Per-job output directories (auto-created)
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the dashboard UI |
| `POST` | `/api/jobs` | Create a new analysis job |
| `GET` | `/api/jobs` | List all jobs (newest first) |
| `GET` | `/api/jobs/{id}` | Get a single job with results and token summary |
| `DELETE` | `/api/jobs/{id}` | Delete a job and its output files |
| `GET` | `/api/jobs/{id}/frames` | Get Phase 2 verification frames (base64) |
| `POST` | `/api/extract-frame` | Extract a video frame for ROI selection |
| `POST` | `/api/extract-timestamp` | Run OCR on a frame region to extract recording timestamp |
| `POST` | `/api/shutdown` | Gracefully shut down the server |

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or directly
python main.py
```

Configure your API keys in `webapp/.env`:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
PHASE1_MODEL_NAME=gpt-4o-mini
PHASE2_MODEL_NAME=gemini-2.5-pro
```

---

## Pending: Production Readiness

The following items are **not yet implemented** and are required before this application can be deployed to a production environment:

### 🔲 Database Integration (Cloud / Persistent)
- The current database is a **local SQLite file** (`jobs.db`), which is not suitable for production deployment (no concurrent writes, lost on container restart, not shared across instances).
- **Required**: Migrate to a production-grade relational database such as **PostgreSQL** (recommended) or **MySQL**.
  - Update `DATABASE_URL` in `.env` to point to the cloud database.
  - Replace `sqlite`-specific SQLAlchemy connect args with a PostgreSQL-compatible engine config.
  - Consider using **Alembic** for schema migrations instead of the custom `_migrate.py` script.
  - Optionally: use a managed cloud DB service (e.g. **AWS RDS**, **Supabase**, **PlanetScale**, **Neon**, **Railway Postgres**).

### 🔲 Public Domain / Cloud Hosting
- Currently the app runs **only on localhost** (port 8000). It needs to be deployed to a public server to be accessible by the team.
- **Required tasks**:
  - Choose a hosting platform (e.g. **Railway**, **Render**, **AWS EC2/ECS**, **Google Cloud Run**, **Azure App Service**, or a bare VPS).
  - Containerize the application with **Docker** — create a `Dockerfile` and optionally a `docker-compose.yml` for local parity.
  - Set up a **reverse proxy** (e.g. Nginx or Caddy) in front of Uvicorn for HTTPS termination and static file serving.
  - Obtain a **domain name** and configure **SSL/TLS** certificates (e.g. via Let's Encrypt / Certbot).
  - Manage secrets securely via the platform's **environment variable / secrets manager** (not `.env` files committed to version control).
  - Implement **persistent file storage** for uploaded videos and results — a local `uploads/` and `results/` folder will not survive container restarts; use a networked storage solution (e.g. **AWS S3**, **Google Cloud Storage**, **Azure Blob Storage**).
  - Add **authentication** (the current app has no login/auth layer) to prevent unauthorized access to potentially sensitive kitchen video footage.
  - Configure **process management** (e.g. Gunicorn + Uvicorn workers, or a supervisor) and a proper production ASGI server setup.
  - Set up **health checks**, **logging aggregation** (e.g. to a centralized log platform), and **monitoring/alerting**.
