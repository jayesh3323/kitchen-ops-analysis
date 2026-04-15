# Architecture Plan: Hugging Face Spaces + Firebase (Zero-Cost MVP)

This document outlines the architecture and implementation steps to move the **Kitchen Ops Dashboard** from a local setup to a cloud-persistent, zero-cost production environment.

---

## 0. Project Status: Current Progress
The project has reached a stable local MVP state. Below is a summary of what has been implemented and is currently being used:

*   **Multi-Agent Pipelines:** Specialized logic for **Pork Weighing**, **Ramen Plating**, and **Service Time** analysis.
*   **Two-Phase Analysis:** 
    *   **Phase 1:** High-speed event detection using `gpt-5-mini`.
    *   **Phase 2:** High-fidelity verification using `gemini-pro`.
*   **Functional Dashboard:** A FastAPI-based web interface allowing users to upload videos, select ROIs, and view a real-time job history.
*   **Background Worker:** A persistent Python thread (`worker.py`) that handles long-running video processing independently of the web request.
*   **Local Persistence:** Data is currently stored in a local `jobs.db` (SQLite), and videos/frames are stored in local `uploads/` and `results/` folders.
*   **ROI Tooling:** Custom OpenCV integration for interactive region selection on video frames.

---

## 1. Core Architecture Overview (Cloud Migration)

To achieve a zero-cost, always-on setup, we are migrating to the following stack:

| Component | Technology | Role | Cost |
| :--- | :--- | :--- | :--- |
| **Compute / Host** | **Hugging Face Spaces** | Runs FastAPI server + Background Python worker thread. Provides **16GB RAM**. | $0 |
| **Database** | **Firebase Firestore** | Stores Job metadata, analysis results, and live progress updates. | $0 |
| **Blob Storage** | **Firebase Storage (GCS)** | Stores uploaded videos (temp) and verification frames (permanent). | $0 |
| **Authentication** | **Firebase Auth** | Secures the dashboard with Email/Password or Google login. | $0 |
| **Observability** | **Langfuse** | Tracks LLM token usage, traces, and **Centralized Prompts**. | $0 |

---

## 2. Platform Integration Details

### Hugging Face Spaces (The Engine)
*   **Deployment Method:** **Docker**. We will use a `Dockerfile` to install `ffmpeg` and OpenCV dependencies.
*   **Persistence:** HF Spaces storage is **ephemeral**. All permanent data must flow to Firebase.
*   **Worker Thread:** The `worker.py` logic remains inside the container, leveraging the 16GB RAM for heavy video extraction.

### Firebase (The Brain & Memory)
*   **Firestore:** Real-time listeners (`onSnapshot`) in the frontend will provide instant UI updates when a job status changes.
*   **Storage:** Videos are uploaded directly from the browser to GCS, saving bandwidth and preventing server-side upload timeouts.

### Langfuse (The Monitor)
*   **Tracing:** Every job log and LLM call is recorded for cost tracking.
*   **Prompt Management:** (See Section 5)

---

## 3. Required Code Changes

### A. Database Layer (`webapp/database.py`)
*   Replace SQLAlchemy/SQLite code with `firebase-admin` Firestore calls.
*   Initialize Firestore collections for `jobs` and `results`.

### B. Background Worker (`webapp/worker.py`)
*   Modify the loop to listen to Firestore `queued` jobs.
*   Download videos from Firebase Storage to `/tmp/`, process them, then upload result frames back to Storage.

### C. Frontend (`webapp/app.js`)
*   Integrate Firebase Auth for secure login.
*   Implement direct-to-storage video uploads.
*   Replace API polling with Firestore listeners.

---

## 4. Implementation Step-by-Step

### Step 1: Firebase Configuration
1.  Create a Firebase project.
2.  Enable Firestore, Storage, and Auth.
3.  Add the `serviceAccountKey.json` as a secret in Hugging Face.

### Step 2: Dockerization
Create the `Dockerfile`:
```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 5. Langfuse Prompt Management

To allow rapid iteration of AI logic without redeploying code, all agent prompts will move to Langfuse.

### How it will work:
1.  **Register Prompts:** We will upload `PHASE1_SYSTEM_PROMPT` and `PHASE2_SYSTEM_PROMPT` for each agent (Pork, Plating, Serve Time) into the Langfuse Prompt Registry.
2.  **Dynamic Fetching:** Inside the Python code, we will replace the hardcoded strings with:
    ```python
    # Fetch latest prompt from Langfuse
    prompt = langfuse.get_prompt("pork-weighing-phase1")
    system_prompt = prompt.compile(context=batch_context)
    ```
3.  **Real-time Tuning:** If the AI starts hallucinating, we can tweak the prompt in the Langfuse UI. The changes take effect **immediately** for the next job.

---

## 6. Summary of Benefits (The "Zero-Cost" Win)
1.  **High RAM (16GB):** Enough power for OpenCV without crashing.
2.  **Data Safety:** Firebase stores everything permanently; HF Space restarts don't lose data.
3.  **Live Workspace:** Multi-user sync via Firestore listeners.
4.  **Prompt Agility:** Tune AI logic in real-time via Langfuse without Docker rebuilds.
