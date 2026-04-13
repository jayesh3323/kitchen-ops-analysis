# Kitchen Ops — Cloud Integration Architecture Plan (MVP)

> **Scope**: Low-traffic internal dashboard · Fast to ship · Uses existing Firebase / Langfuse familiarity  
> **Goal**: Jobs persist and run in the background on the server even when the browser is closed; all results, frames, and progress are persisted in the cloud and retrieved whenever the UI is reopened.

---

## TL;DR — Recommended Stack

| Concern | Chosen Service | Why |
|---|---|---|
| **Auth** | Firebase Authentication | Already used; zero extra cost; simple email/Google login |
| **Job metadata + results DB** | Firebase Firestore | Already used; real-time listeners = live job progress without polling; no SQL migration needed |
| **File storage** (videos, frames) | Firebase Storage (GCS) | Same Firebase project; large file support; generous free tier; direct upload from browser |
| **Background worker** | Current FastAPI server (long-running Python process on a VM/VPS) | Keeps the existing `worker.py` threading model intact; simplest path |
| **Hosting (API server)** | Google Cloud Run OR a simple Linux VPS (Railway / Render / DigitalOcean) | Cloud Run is serverless but needs persistent worker; VPS is simpler for MVP |
| **Frontend hosting** | Firebase Hosting | Free, CDN-backed, already in the Firebase ecosystem |
| **LLM Observability** | Langfuse (cloud) | Already wired in via `.env`; tracks token cost per phase per job |
| **Vector search (future)** | Pinecone | Not needed for MVP; earmark for future semantic search over results |

---

## Why This Stack (Detailed Reasoning)

### 1. Firebase Firestore as the Primary Database

**Current state**: The app uses SQLite (`jobs.db`) with SQLAlchemy ORM for three tables — `Job`, `Result`, `TokenUsage`.

**Problem with SQLite in production**:
- It's a single file on disk. If the server restarts or redeploys, the file is gone unless you have persistent volumes.
- It cannot be accessed by multiple processes or servers simultaneously.
- It can't be read from the browser without going through the backend API.

**Why Firestore over PostgreSQL (Supabase/RDS)?**  
Since you already use Firebase, choosing Firestore means:
- **Zero new vendor**: You already have billing, SDK familiarity, and credentials set up.
- **Real-time listeners**: Firestore's `onSnapshot()` lets the browser subscribe to job document changes. This natively replaces the current `setInterval` polling hack in `app.js` — the UI updates the moment the server writes a new `progress_message` or `status` to the job document. This is the key requirement for "show progress even after the page is reopened."
- **No schema migrations**: Firestore is schema-less. You just write Python dicts. No Alembic, no `_migrate.py`.
- **Generous free tier (Spark plan)**: 1 GB storage, 50k reads/day, 20k writes/day — more than enough for an internal low-traffic dashboard.
- **Side-by-side with SQLAlchemy is removable**: The existing `database.py` SQLAlchemy layer gets replaced with Firebase Admin SDK calls. Not a huge refactor — just swap the session/query calls.

**Firestore Collection Structure**:
```
/jobs/{job_id}/
    video_name, video_path (GCS URI), agent, status,
    progress_message, phase1_count, phase2_count, total_tokens,
    roi_coords, timestamp_region_coords, config_json,
    recording_date, recording_hour, created_at, updated_at, error_message

/jobs/{job_id}/results/{event_id}/
    event_id, start_time, end_time, scale, phase1_reading,
    verified_reading, unit, reading_state, confidence,
    description, real_timestamp, frame_gcs_url

/jobs/{job_id}/token_usage/{phase}/
    phase, prompt_tokens, completion_tokens, total_tokens
```

---

### 2. Firebase Storage for Videos and Frames

**Current state**: Videos go into `webapp/uploads/`, result frames go into `webapp/results/job_X_*/phase2_frames/`.

**Problem**: Both folders live on the server disk. If the container/VM is reprovisioned, everything is lost.

**Why Firebase Storage (GCS under the hood)**:
- Same Firebase project — one set of credentials, one Admin SDK import.
- Supports very large file uploads (video files up to several GB with resumable uploads).
- Framework for access rules is already Firebase Auth-integrated: you can restrict downloads to authenticated users only.
- The frames (PNG/JPEG images) from Phase 2 are currently returned as base64 blobs over the API. Instead, store them in Firebase Storage and return the download URL — the browser loads them directly from GCS CDN, not from your server. This reduces server bandwidth significantly.

**Upload flow for videos**:
```
Browser → Firebase Storage (direct upload, bypasses FastAPI server)
         → After upload complete, browser sends GCS URI to FastAPI POST /api/jobs
         → Worker downloads video from GCS to a local temp path for processing
         → After job completes, worker deletes local temp copy
```
This is critical: it means **your server never needs to handle a 2 GB video upload** — the browser uploads directly to Firebase Storage. The `MAX_UPLOAD_SIZE_MB` limit in the current code becomes irrelevant.

**Frame storage flow**:
```
Worker saves Phase 2 frames to local disk during pipeline run
         → After each frame is saved, upload it to Firebase Storage
         → Store the download URL in the Firestore result document
         → Browser loads frames directly from GCS URL (no base64 API call needed)
```

---

### 3. Background Worker: Keep the Existing Python Thread Model

**Current state**: `worker.py` runs as a daemon thread inside the FastAPI process. It polls Firestore (replacing SQLite) for queued jobs and processes them one at a time.

**The key requirement: "Jobs keep running even if the browser is closed"**

This is already satisfied by the current threading model — the `worker_loop()` thread runs in the Python process independently of any HTTP connection. The browser being closed or disconnected has zero effect on it.

**What needs to change**:
- The worker polls **Firestore** instead of SQLite.
- Progress updates write to the **Firestore job document** instead of local DB.
- When the browser reopens and loads the dashboard, it reads from Firestore — all job states are up-to-date regardless of downtime.

**Why not Celery + Redis?**  
- Celery adds significant complexity (two processes, Redis server, task serialization) for what is currently one sequential background thread.
- For an internal low-traffic MVP where only a few jobs run per day, and they're sequential by design (GPU/CPU-bound pipeline), there is no benefit to Celery's parallelism.
- Firebase Firestore can serve as a lightweight job queue (query for `status == "queued"`, process, update to `"processing"`).

**Why not Cloud Run Jobs / Cloud Functions?**  
- The analysis pipeline can run for 10–30+ minutes per video. Cloud Functions time out at 9 minutes (1st gen) or 60 minutes (2nd gen). Cloud Run Jobs are more suitable but add Docker + GCP complexity.
- For MVP: **a single long-running VM or managed service (Railway/Render) hosting both FastAPI + worker thread is sufficient and simplest.**

---

### 4. Hosting: Railway or Render (VPS-style), Not Pure Serverless

**Why not serverless (Cloud Run, Lambda, etc.)**:
- The video analysis pipeline is CPU-heavy and can run for 10–60 minutes. Serverless platforms time out and don't suit long-running, stateful computation.
- The background thread model requires a **persistent, always-on process**.

**Recommended for MVP**: **Railway** or **Render**

Both offer:
- Simple Git-push deployment (no Docker knowledge required initially).
- Persistent disk (Railway: persistent volumes; Render: disk mounts).
- Free/cheap starting tiers.
- Environment variable management in the dashboard.
- Auto-restart on crash.

Railway is slightly preferred because:
- First-class Python/FastAPI support.
- $5/month hobby plan covers the compute needed.
- Clean environment variable UI that mirrors your `.env` file.

**Server specs for the worker**:  
The pipeline is CPU-bound (OpenCV frame extraction, image encode/decode) and makes API calls (OpenAI / Gemini). It does not need a GPU. A **1 vCPU / 2 GB RAM** instance is sufficient for MVP with sequential job processing.

---

### 5. Firebase Authentication

**Current state**: No authentication. Anyone with the URL can submit jobs.

**Why Firebase Auth**:
- Already in your stack.
- No backend session management needed — Firebase handles JWT tokens.
- The FastAPI backend verifies the Firebase ID token on protected endpoints using `firebase_admin`.
- For an internal tool, email/password auth or Google Sign-In is sufficient.
- Firestore and Firebase Storage security rules can restrict read/write to authenticated users only.

---

### 6. Langfuse for LLM Observability

**Current state**: Langfuse keys are in `.env` but integration is largely commented out or removed from `worker.py`.

**Recommended**: Re-enable Langfuse traces wrapping the pipeline calls.
- Wrap `pipeline.run_phase1()` and `pipeline.run_phase2()` in Langfuse `trace()` spans.
- Each Langfuse trace links to the `job_id` so you can correlate dashboard jobs with LLM call costs.
- Langfuse cloud (free tier: 50k observations/month) is more than enough for MVP.
- This is optional for MVP but very useful for cost tracking since Gemini 2.5 Pro is expensive.

---

### 7. Pinecone — Defer to Post-MVP

Pinecone is a vector database. It's not needed for the current MVP requirements (job tracking, results storage, frame storage). Earmark it for a future feature such as:
- Semantic search over all past detection results ("find all jobs where a scale reading was under 180g").
- Embedding-based anomaly detection ("flag jobs whose results are statistically unusual").

---

## Full Architecture Diagram (Text)

```
┌──────────────────────────────────────────────────────────────────┐
│                        BROWSER (Dashboard)                        │
│                                                                    │
│  Firebase JS SDK                                                   │
│  ├── Firebase Auth      → login / JWT token for all requests      │
│  ├── Firestore SDK      → onSnapshot() live job status updates    │
│  └── Firebase Storage   → direct video upload (resumable)        │
│                           direct frame download (GCS URLs)        │
│                                                                    │
│  FastAPI calls (fetch)  → POST /api/jobs (send GCS video URI)    │
│                         → POST /api/extract-frame                 │
│                         → POST /api/extract-timestamp             │
│                         → DELETE /api/jobs/{id}                   │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTPS
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│               FASTAPI SERVER + BACKGROUND WORKER                  │
│               (Railway / Render — always-on VM)                   │
│                                                                    │
│  main.py (FastAPI)                                                 │
│  ├── Validates Firebase JWT on protected endpoints                 │
│  ├── Creates job document in Firestore                             │
│  └── Returns job_id to browser                                    │
│                                                                    │
│  worker.py (background thread — runs even when browser closed)    │
│  ├── Polls Firestore for status == "queued"                       │
│  ├── Downloads video from Firebase Storage → local /tmp           │
│  ├── Runs pipeline_adapter → run_phase1() → run_phase2()         │
│  ├── Writes progress_message to Firestore in real time            │
│  ├── Uploads Phase 2 frames → Firebase Storage                    │
│  ├── Writes results (events) as subcollection in Firestore        │
│  ├── Writes token_usage subcollection in Firestore                │
│  └── Updates job status = "completed" / "failed" in Firestore    │
└──────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐
  │   Firestore   │  │ Firebase      │  │    Langfuse        │
  │               │  │ Storage       │  │    (cloud)         │
  │  /jobs/       │  │               │  │                    │
  │  /jobs/*/     │  │ /videos/      │  │  LLM traces        │
  │    results/   │  │ /frames/      │  │  per job_id        │
  │  token_usage/ │  │               │  │  token costs       │
  └───────────────┘  └───────────────┘  └───────────────────┘
                              │
                     ┌───────────────┐
                     │   OpenAI GPT  │  ← Phase 1 (event detection)
                     │   Google      │  ← Phase 2 (verification)
                     │   Gemini      │
                     └───────────────┘
```

---

## Implementation Plan (Phased, Ordered by Priority)

### Phase A — Firestore replaces SQLite (Most impactful, do first)

1. Add `firebase-admin` to `requirements.txt`.
2. Create a Firebase project (or reuse existing) → generate a service account JSON key.
3. Rewrite `database.py`:
   - Remove SQLAlchemy, replace with `firebase_admin.firestore` client.
   - Port `Job.to_dict()` / `Result.to_dict()` to plain Python dict helpers.
4. Update `worker.py`:
   - Replace `SessionLocal()` queries with Firestore `collection("jobs").where("status", "==", "queued")`.
   - Replace `_update_job()` with `doc_ref.update({...})`.
5. Update `main.py` API endpoints to read/write Firestore.
6. **Frontend change**: Replace `setInterval` polling in `app.js` with Firestore `onSnapshot()` listener. The job row updates live the instant the server writes to Firestore.

> ✅ After this phase: jobs persist forever, worker runs independently, browser gets live updates.

---

### Phase B — Firebase Storage replaces local disk

1. Add Firebase Storage SDK to both Python (Admin SDK) and browser (JS SDK).
2. **Video upload**: Change the upload flow in `app.js` → upload directly to Firebase Storage, get the GCS URI, pass it to `/api/jobs` instead of sending the file to FastAPI.
3. **Worker download**: At job start, `worker.py` downloads the video from GCS to `/tmp/job_{id}_video.mp4` for local processing. Delete after job completion.
4. **Frame upload**: After each Phase 2 frame is saved locally, upload it to `gs://your-bucket/frames/job_{id}/frame_N.png` → store the public download URL in the Firestore result document.
5. **Frontend**: Load frames from GCS download URLs directly in `<img>` tags — no more base64 API endpoint needed.

> ✅ After this phase: all files are cloud-persistent. Server disk is ephemeral/temp only.

---

### Phase C — Firebase Auth

1. Add Firebase SDK to `index.html` (JS).
2. Add a login page / modal (email/password or Google Sign-In).
3. After login, pass the Firebase ID Token as `Authorization: Bearer <token>` header on all `fetch()` calls.
4. In FastAPI: add a dependency that validates the token with `firebase_admin.auth.verify_id_token()`.
5. Set Firestore security rules to `request.auth != null`.

> ✅ After this phase: the dashboard is protected from unauthorized access.

---

### Phase D — Deploy to Railway

1. Create a `Dockerfile` (or use Railway's Python buildpack — no Docker needed):
   ```
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
2. Push to a GitHub repo → connect Railway to the repo.
3. Set all `.env` values as Railway environment variables (never commit `.env` to git).
4. Point a custom domain at the Railway service.
5. Railway auto-provisions HTTPS.

> ✅ After this phase: the dashboard is publicly accessible at `https://your-domain.com`.

---

### Phase E — Re-enable Langfuse (optional but recommended)

1. Wrap `run_phase1()` and `run_phase2()` calls in `pipeline_adapter.py` with Langfuse `trace()`/`span()`.
2. Store the `job_id` as Langfuse trace metadata so you can link dashboard jobs to LLM cost reports.

---

## What Does NOT Need to Change

| Current code | Status |
|---|---|
| `main.py` FastAPI endpoints | Keep structure, swap DB calls |
| `worker.py` threading model | Keep exactly — just swap storage backend |
| `pipeline_adapter.py` | No change needed |
| `agents/*.py` scripts | No change needed |
| `config.py` | Add Firebase config vars |
| `static/style.css` | No change |
| `templates/index.html` | Add Firebase JS SDK + Auth UI |
| `static/app.js` | Replace polling with `onSnapshot`, add direct Storage upload |

---

## Cost Estimate (MVP, Low Traffic)

| Service | Tier | Estimated Cost |
|---|---|---|
| Firebase (Firestore + Storage + Auth + Hosting) | Spark (free) or Blaze (pay-as-you-go) | **~$0–$5/month** for low traffic |
| Railway (FastAPI + worker server) | Hobby plan | **~$5/month** |
| Langfuse | Cloud free tier | **$0** |
| Domain name | Any registrar | **~$10–15/year** |
| OpenAI + Google Gemini API | Pay per use | Depends on usage (main cost) |

**Total infra cost: ~$10–15/month** (excluding LLM API costs).

---

## Summary

The recommended MVP architecture is:

> **Firebase (Firestore + Storage + Auth + Hosting) + Railway (FastAPI server + Python background worker thread) + Langfuse** 

This works because:
1. **Firebase is already your stack** — no new vendor learning curve.
2. **Firestore real-time listeners** eliminate the polling hack and give true live progress updates across browser sessions and page refreshes.
3. **Firebase Storage** handles large video files without going through your server, and stores frames permanently.
4. **The existing background worker model survives unchanged** — the thread runs on Railway's always-on VM and processes jobs even when the browser is closed.
5. **It's the smallest possible changeset** — only `database.py`, `worker.py`, the API calls in `main.py`, and the JS frontend need to change. All pipeline code stays intact.
6. **Cost is minimal** for internal low-traffic use.
