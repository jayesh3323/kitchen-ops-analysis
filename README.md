# 🥩 Pork Weighing Analysis — Web Application

A web-based deployment of the Pork Weighing Analysis Pipeline with background processing, Langfuse observability, and persistent results.

## Architecture

```
Browser (Dashboard) → FastAPI Server → Background Worker Thread
                           ↓                    ↓
                      SQLite DB          PorkWeighingPipeline
                                               ↓
                                         Langfuse Traces
```

**No Redis or Docker required!** The worker uses a pure-Python SQLite-polling thread that runs embedded in the FastAPI server.

## Prerequisites

1. **Python 3.10+** with existing pipeline dependencies installed
2. **FFmpeg** — Already required by the parent pipeline

## Quick Start

### 1. Install Dependencies

```bash
cd webapp
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template and fill in your Langfuse keys (optional)
cp .env.example .env
# Edit .env with your LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

> **Note:** API keys (OpenAI, Google) are inherited from the parent `.env` file. You don't need to duplicate them.

### 3. Start the Server

```bash
cd webapp
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --port 8000
```

The background worker starts automatically as a daemon thread — no separate terminal needed!

### 4. Open Dashboard

Visit **http://localhost:8000** in your browser.

## Usage

1. **Upload a video** (up to 500MB) or enter a server file path for larger files
2. **Select ROI** — Click "Select ROI" to draw a rectangle on a video frame identifying the scale display area
3. **Configure** — Set FPS, toggle Phase 2 verification
4. **Submit** — Click "Start Analysis" to queue the job
5. **Monitor** — The jobs table auto-refreshes every 5 seconds showing progress
6. **Close browser** — Processing continues in the background!
7. **Return later** — See completed results and click "View in Langfuse" for detailed traces

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard page |
| `POST` | `/api/jobs` | Create new analysis job |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/jobs/{id}` | Get job details + results |
| `DELETE` | `/api/jobs/{id}` | Delete a job |
| `POST` | `/api/extract-frame` | Extract a frame for ROI selection |

## Standalone Worker (Optional)

If you prefer to run the worker separately (e.g., on a different machine):

```bash
cd webapp
python worker.py
```

## Project Structure

```
webapp/
├── main.py                  # FastAPI server + embedded worker
├── worker.py                # Background worker (SQLite-polling)
├── config.py                # Centralized configuration
├── database.py              # SQLAlchemy models
├── langfuse_integration.py  # Langfuse tracing
├── pipeline_adapter.py      # Headless pipeline wrapper
├── requirements.txt         # Dependencies
├── .env                     # Local configuration
├── .env.example             # Configuration template
├── templates/
│   └── index.html           # Dashboard HTML
└── static/
    ├── style.css            # Dark theme styling
    └── app.js               # Client-side logic
```
