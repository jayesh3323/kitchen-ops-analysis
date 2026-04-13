# Research Findings & Implementation Tracker

> **How to use:** The "Result" column is for manual updates after testing. Fill in "✅ Working", "❌ Failed", "⚠️ Partial", or notes on observed accuracy change.

---

## Implementations Made

| # | Change | File(s) | Description | Result |
|---|--------|---------|-------------|--------|
| I1 | CLAHE preprocessing on pork ROI crop | `agents/pork_weighing_compliance.py` | Applied CLAHE (clipLimit=2.0, tileGridSize=8×8) on L channel (LAB space) after ROI crop, before upscale. Improves LCD digit contrast without color distortion. | ❌ Failed |
| I2 | Sequential ROI & Display VLM Pipeline | `auto_roi.py` | Two-step sequential flow (ROI first, then Displays using ROI as context). Dramatically reduces false-positive "food circles".| |
| I3 | Red Circle Visual Prompting | All 5 agents | **Red Circle Visual Prompting** — draw `cv2.circle()` on the exact region the VLM must attend to (red outperforms all other colors/shapes). 72–128% relative improvement on localization (keypoint: 42.2%→7s% PCP) | ✅ Working |
| I4 | Rolling temporal context: last 3 batch summaries | All 5 agents | **Rolling temporal context: last 3 batch summaries** — replace `context_history[-1:]` with `context_history[-3:]` in Phase 1 loop. 15–20% better recall on multi-step events | No change in accuracy, but tokens saved |
| I5 | Grounded description pre-prompt | All 5 agents | **Grounded description pre-prompt** — instruct model to describe visual evidence before classifying (add block to PHASE1_SYSTEM_PROMPT). Visual Chain-of-Thought prompting | ✅ Working |
| I6 | Lower confidence thresholds | All 5 agents | **Lower confidence thresholds** (safe after voting active): pork 0.60→0.50, plating 0.70→0.55, serve 0.70→0.55, noodle 0.80→0.65, bowl 0.80→0.65 | Works well for pork weighing task |
| I7 | Few-shot reference images in Phase 1 prompts | All 5 agents | **Few-shot reference images in Phase 1 prompts** — 2–3 annotated JPEG crops per agent inline as base64. +15–25% accuracy; saturates at 3 examples | ✅ Working |
| I8 | Real-ESRGAN super-resolution on digit sub-crop |	pork_weighing_compliance.py |	Applied 4× Real-ESRGAN SR on a tight sub-crop around each detected scale display (1.5× padded bounding box from display_circles), then Lanczos-downsampled the SR output back to the original patch size before pasting in-place. |	✅ Working


---

## CV Preprocessing Research (Image-Level Techniques)

These techniques operate on **pixel data before VLM encoding** — no architecture change, no prompt change.

| # | Technique | Research Source | Expected Gain | Agents | Status | Result |
|---|-----------|----------------|---------------|--------|--------|--------|
| CV1 | **Red Circle Visual Prompting** — draw `cv2.circle()` on the exact region the VLM must attend to (red outperforms all other colors/shapes) | [ICCV 2023 — "What does CLIP know about a red circle?"](https://openaccess.thecvf.com/content/ICCV2023/papers/Shtedritski_What_does_CLIP_know_about_a_red_circle_Visual_prompt_ICCV_2023_paper.pdf) | 72–128% relative improvement on localization (keypoint: 42.2%→72% PCK) | All 5 (each circles its key region) | **Implemented (I2)** | ✅ Working |
| CV3 | **Optical Flow Color Overlay** — compute Farneback dense flow between consecutive frames, convert to HSV color map, blend onto RGB frame before encoding | [MDPI Entropy 2022](https://www.mdpi.com/1099-4300/24/7/939); [RPEFlow ICCV 2023](https://arxiv.org/html/2309.15082) | Explicit motion direction cues; VLM can read flow arrows to describe rotation direction | noodle_rotation, plating_time | Not implemented | |
| CV5 | **Real-ESRGAN Super-Resolution** — apply 2× SR on tight digit sub-crop (bottom 25% of ROI) before encoding | [ICCV 2021 Workshop](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf); [OCR accuracy paper](https://www.oajaiml.com/archive/enhancing-ocr-performance-through-super-resolution-reconstruction-using-ssae-real-esrgan) | +10.27% PSNR, +10.38% SSIM on text recognition | pork_weighing only | **Implemented (I8)** | ✅ Working |
| CV8 | **CLAHE on pork ROI crop** (LAB L-channel, clipLimit=2.0, tileGridSize=8×8) — applied before upscale | Standard OpenCV technique; ICCV 2021 Real-ESRGAN paper notes contrast preprocessing as prerequisite | Improves LCD digit legibility; complements super-resolution | pork_weighing | **Implemented (I1)** | ❌ Failed |

---

## Pipeline / Architecture Research

Changes to prompting strategy, model selection, sampling, or multi-pass logic.

| # | Technique | Research Source | Expected Gain | Agents | Status | Result |
|---|-----------|----------------|---------------|--------|--------|--------|
| A1 | **Upgrade Phase 1 model: GPT-5-mini → o4-mini** — o4-mini integrates images into reasoning chain; add `max_completion_tokens=4096`, `reasoning_effort="medium"` | [arXiv 2509.07969](https://arxiv.org/html/2509.07969v1); [Roboflow o4-mini blog](https://blog.roboflow.com/openai-o3-and-o4-mini/) | +13–20pp on VisualProbe-Hard vs GPT-5-mini | pork_weighing_compliance.py | ❌ Failed | |
| A2 | **Increase frame resolution: 640→1024px (pork), 448→1024px (other 4)** | [Token-Efficient VLM ICCV 2025](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.pdf) | 15–25% reduction in false negatives for fine-grained actions | pork_weighing_compliance.py | ❌ Failed | |
| B1 | **Self-consistency multi-pass voting** — 3 API calls per batch at T=0.8; keep detections where ≥2/3 passes agree (temporal overlap ≤2s) | [arXiv 2503.20472](https://arxiv.org/html/2503.20472) | 20–30% reduction in false negatives; real events with 0.5–0.7 confidence survive voting | All 5 | Not implemented | |
| B2 | **Rolling temporal context: last 3 batch summaries** — replace `context_history[-1:]` with `context_history[-3:]` in Phase 1 loop | [Temporal CoT arXiv 2507.02001](https://arxiv.org/html/2507.02001v1); [VideoScan arXiv 2503.09387](https://arxiv.org/html/2503.09387v2) | 10–20% better recall on multi-step events | All 5 | Implemented | No change in accuracy, but tokens saved |
| B3 | **Grounded description pre-prompt** — instruct model to describe visual evidence before classifying (add block to PHASE1_SYSTEM_PROMPT) | [Visual CoT arXiv 2403.16999](https://arxiv.org/html/2403.16999v1); [Chain-of-Visual-Thought arXiv 2511.19418](https://arxiv.org/html/2511.19418v2) | Accuracy improvement; only deploy after o4-mini (degrades GPT-5-mini per arXiv 2603.16728) | All 5 | Implemented | Working |
| C1 | **Lower confidence thresholds** (safe after voting active): pork 0.60→0.50, plating 0.70→0.55, serve 0.70→0.55, noodle 0.80→0.65, bowl 0.80→0.65 | Internal analysis + B1 voting paper | Recover real events currently filtered out; false-positive protection handled by voting | All 5 | Implemented | Works well for pork weighing task |
| C2 | **Few-shot reference images in Phase 1 prompts** — 2–3 annotated JPEG crops per agent inline as base64 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/22b2067b8f680812624032025864c5a1-Paper-Datasets_and_Benchmarks_Track.pdf) | +15–25% accuracy; saturates at 3 examples (1 clear positive, 1 partial/occluded, 1 low-light) | All 5 | Implemented | Working |
| G2 | **JSON schema enforcement** — pass `response_format={"type":"json_schema",...}` to GPT; `response_schema` to Gemini | OpenAI + Google API docs | Eliminates parse failures; ~5% retry token reduction | All 5 | Not implemented | |
| G4 | **Motion-based chunk pre-screening** — mean absolute frame-diff within ROI per chunk; skip if below threshold | [DyToK arXiv 2512.06866] | 30–50% fewer Phase 1 API calls on idle periods | All 5 | Not implemented | |

---

## Phase 2 Model Alternatives (Pork Weighing OCR)

Current Phase 2 model: `gemini-2.5-pro`. Alternatives evaluated for LCD digit OCR accuracy.

| Model | Drop-in? | Code change needed | OCR strength | Notes |
|-------|----------|--------------------|--------------|-------|
| `gemini-2.5-flash` | Yes | Change `AGENT_PHASE2_MODEL_NAME` only | Strong; 3–5× cheaper than Pro | Best cost/accuracy tradeoff for clear digits |
| `gpt-4o` | Near-drop-in | Use existing `openai_client`; reformat Gemini→OpenAI call | Very strong digit disambiguation | Best alternative if Gemini quota is a concern |
| `claude-opus-4-6` | No | Add Anthropic client + new Phase 2 branch | Best digit OCR (Claude 4.6 is top on text tasks) | Highest accuracy; requires new SDK integration |
| PaddleOCR (hybrid) | No | Add `paddlepaddle` dependency; run locally before VLM call | Best raw OCR; deterministic | Hybrid: OCR reads digits, VLM confirms context/events |

---

## Infrastructure Fixes (Queued)

| # | Fix | File | Description | Status |
|---|-----|------|-------------|--------|
| F1 | Eliminate 30s job pickup lag | `worker.py` | Replace `time.sleep(30)` with `job_cache.wait_for_queued(timeout=30)` | Not implemented |
| F2 | Path traversal security fix | `main.py` | Validate `video_path` is inside `UPLOAD_DIR` before `os.path.exists()` | Not implemented |
| F3 | File type validation on upload | `main.py` | Reject non-video extensions (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) | Not implemented |
| F4 | Config validation at startup | `config.py` | Fail fast if `OPENAI_API_KEY`/`GOOGLE_API_KEY` missing or `FPS <= 0` | Not implemented |
| F5 | Push confidence filter to DB query | `main.py`, `database.py` | Add `WHERE confidence >= threshold` to SQL instead of filtering in Python | Not implemented |
| F6 | Job cancellation endpoint | `main.py`, `worker.py` | `POST /api/jobs/{id}/cancel`; sets cancel flag checked in progress callback | Not implemented |
| F7 | Parallel job processing | `worker.py` | `ThreadPoolExecutor(max_workers=2)` to process 2 jobs concurrently | Not implemented |
| F8 | Partial result recovery | `worker.py`, `database.py` | Save Phase 1 results immediately; on retry skip Phase 1 if already saved | Not implemented |
