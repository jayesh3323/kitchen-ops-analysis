# Research Findings & Implementation Tracker

> **How to use:** The "Result" column is for manual updates after testing. Fill in "✅ Working", "❌ Failed", "⚠️ Partial", or notes on observed accuracy change.

---

## Implementations Made

| # | Change | File(s) | Description | Result |
|---|--------|---------|-------------|--------|
| I1 | CLAHE preprocessing on pork ROI crop | `agents/pork_weighing_compliance.py` | Applied CLAHE (clipLimit=2.0, tileGridSize=8×8) on L channel (LAB space) after ROI crop, before upscale. Improves LCD digit contrast without color distortion. | ❌ Failed |
| I2 | Sequential ROI & Display VLM Pipeline | `auto_roi.py` | Two-step sequential flow (ROI first, then Displays using ROI as context). Dramatically reduces false-positive "food circles". |  |

---
# Research and Changes Documentation

## 2026-04-07: Optimizing Pork Weighing ROI & Display Detection

### Objective
Improve the accuracy of scale ROI detection and digital display (red circle) localization for the pork weighing compliance pipeline.

### Changes & Research Observations

#### 1. Unified vs. Sequential VLM Detection
- **Initial Approach:** A single VLM call to GPT-4o-mini attempted to find both the Green ROI and Red Display Circles.
- **Problem:** When the image contained multiple similar objects (bowls, food), the VLM often confused food items for display panels, especially if the initial ROI wasn't tight enough.
- **Solution:** Switched to a **Sequential Two-Step Detection** flow.
    - **Step 1:** Detect the primary weighing scale apparatus (ROI) on the full frame.
    - **Step 2:** Provide the detected ROI coordinates as context to a second VLM call on the same frame specifically focused on locating display panels within that ROI.
- **Benefit:** By forcing the model to explicitly identify the ROI first and then using that ROI as a spatial constraint in the next request, the rate of "food circles" (misidentified displays) dropped significantly.

#### 2. Visual Prompting Clarity
- **Prompt Refinement:** Added "green background" to the description of digital displays to help the model distinguish between LCD readouts and bowl interiors.
- **Precision Instructions:** Instructed the model to provide **"VERY TIGHT"** bounding boxes for both regions to avoid capturing surrounding noise.

#### 3. Spatial Filtering
- Implemented a server-side **Sanity Check** that filters out any display coordinates detected far outside the scale ROI. This acts as a safety gate for cases where the VLM might still hallucinate a display on a faraway object.

#### 4. UI Rendering Adjustments
- Reduced the circle radius padding from **20% to 5%**.
- Synchronized the `_AGENT_MARGIN` in `auto_roi.py` and the pipeline radius logic in `pork_weighing_compliance.py` to ensure high precision in the final visual verification.

### Results
The sequential detection flow provides a much higher "hit rate" for the actual digital screens. The tight focus on the scale apparatus ensures that visual prompts in Phase 1 and Phase 2 are centered exactly on the weight readout, leading to more reliable OCR results.

---

## CV Preprocessing Research (Image-Level Techniques)

These techniques operate on **pixel data before VLM encoding** — no architecture change, no prompt change.

| # | Technique | Research Source | Expected Gain | Agents | Status | Result |
|---|-----------|----------------|---------------|--------|--------|--------|
| CV1 | **Red Circle Visual Prompting** — draw `cv2.circle()` on the exact region the VLM must attend to (red outperforms all other colors/shapes) | [ICCV 2023 — "What does CLIP know about a red circle?"](https://openaccess.thecvf.com/content/ICCV2023/papers/Shtedritski_What_does_CLIP_know_about_a_red_circle_Visual_prompt_ICCV_2023_paper.pdf) | 72–128% relative improvement on localization (keypoint: 42.2%→72% PCK) | All 5 (each circles its key region) | **Implemented (I5-I8)** |  |
| CV2 | **Set-of-Mark (SoM) Prompting** — overlay numbered bounding-box labels on semantic regions; prompt references regions by number | [arXiv 2310.11441](https://arxiv.org/abs/2310.11441) | Competitive with fine-tuned SOTA on RefCOCOg zero-shot | plating_time, avg_serve_time, bowl_completion_rate | Not implemented | |
| CV3 | **Optical Flow Color Overlay** — compute Farneback dense flow between consecutive frames, convert to HSV color map, blend onto RGB frame before encoding | [MDPI Entropy 2022](https://www.mdpi.com/1099-4300/24/7/939); [RPEFlow ICCV 2023](https://arxiv.org/html/2309.15082) | Explicit motion direction cues; VLM can read flow arrows to describe rotation direction | noodle_rotation, plating_time | Not implemented | |
| CV4 | **Laplacian Variance Frame Filtering** — compute `cv2.Laplacian().var()` per frame; skip frames below threshold (150), use nearest sharp neighbor | [arXiv 2504.13690](https://arxiv.org/html/2504.13690v2); [arXiv 2603.06148](https://arxiv.org/abs/2603.06148) | 5–10 accuracy point recovery (blur causes 8–15pp drop; TextVQA: 57.5%→45% under blur) | All 5 (in `extract_frames()`) | Not implemented | |
| CV5 | **Real-ESRGAN Super-Resolution** — apply 2× SR on tight digit sub-crop (bottom 25% of ROI) before encoding | [ICCV 2021 Workshop](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf); [OCR accuracy paper](https://www.oajaiml.com/archive/enhancing-ocr-performance-through-super-resolution-reconstruction-using-ssae-real-esrgan) | +10.27% PSNR, +10.38% SSIM on text recognition | pork_weighing only | Not implemented | |
| CV6 | **NAFNet Motion Deblurring** — run NAFNet on frames where Laplacian is low AND optical flow magnitude is high (motion blur, not defocus) | [arXiv 2204.04676](https://arxiv.org/abs/2204.04676) | 70ms/frame GPU (SA-NAFNet variant); restores sharp noodle bundle shape | noodle_rotation | Not implemented | |
| CV7 | **Attention Prompting on Image** — compute CLIP text-query attention heatmap, blend Jet colormap onto frame before encoding | [ECCV 2024 — arXiv 2409.17143](https://arxiv.org/html/2409.17143v1) | +11.6% GPT-4V VisWiz, +8.3% Gemini VisWiz, +3.8% LLaVA-1.5 MM-Vet | All 5 (query-specific per agent) | Not implemented | |
| CV8 | **CLAHE on pork ROI crop** (LAB L-channel, clipLimit=2.0, tileGridSize=8×8) — applied before upscale | Standard OpenCV technique; ICCV 2021 Real-ESRGAN paper notes contrast preprocessing as prerequisite | Improves LCD digit legibility; complements super-resolution | pork_weighing | **Implemented (I1)** | |

---

## Pipeline / Architecture Research

Changes to prompting strategy, model selection, sampling, or multi-pass logic.

| # | Technique | Research Source | Expected Gain | Agents | Status | Result |
|---|-----------|----------------|---------------|--------|--------|--------|
| A1 | **Upgrade Phase 1 model: GPT-5-mini → o4-mini** — o4-mini integrates images into reasoning chain; add `max_completion_tokens=4096`, `reasoning_effort="medium"` | [arXiv 2509.07969](https://arxiv.org/html/2509.07969v1); [Roboflow o4-mini blog](https://blog.roboflow.com/openai-o3-and-o4-mini/) | +13–20pp on VisualProbe-Hard vs GPT-5-mini | pork_weighing_compliance.py | ❌ Failed | |
| A2 | **Increase frame resolution: 640→1024px (pork), 448→1024px (other 4)** | [Token-Efficient VLM ICCV 2025](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.pdf) | 15–25% reduction in false negatives for fine-grained actions | pork_weighing_compliance.py | ❌ Failed | |
| B1 | **Self-consistency multi-pass voting** — 3 API calls per batch at T=0.8; keep detections where ≥2/3 passes agree (temporal overlap ≤2s) | [arXiv 2503.20472](https://arxiv.org/html/2503.20472) | 20–30% reduction in false negatives; real events with 0.5–0.7 confidence survive voting | All 5 | Not implemented | |
| B2 | **Rolling temporal context: last 3 batch summaries** — replace `context_history[-1:]` with `context_history[-3:]` in Phase 1 loop | [Temporal CoT arXiv 2507.02001](https://arxiv.org/html/2507.02001v1); [VideoScan arXiv 2503.09387](https://arxiv.org/html/2503.09387v2) | 10–20% better recall on multi-step events | All 5 | Not implemented | |
| B3 | **Grounded description pre-prompt** — instruct model to describe visual evidence before classifying (add block to PHASE1_SYSTEM_PROMPT) | [Visual CoT arXiv 2403.16999](https://arxiv.org/html/2403.16999v1); [Chain-of-Visual-Thought arXiv 2511.19418](https://arxiv.org/html/2511.19418v2) | Accuracy improvement; only deploy after o4-mini (degrades GPT-5-mini per arXiv 2603.16728) | All 5 | Not implemented | |
| C1 | **Lower confidence thresholds** (safe after voting active): pork 0.60→0.50, plating 0.70→0.55, serve 0.70→0.55, noodle 0.80→0.65, bowl 0.80→0.65 | Internal analysis + B1 voting paper | Recover real events currently filtered out; false-positive protection handled by voting | All 5 | Not implemented | |
| C2 | **Few-shot reference images in Phase 1 prompts** — 2–3 annotated JPEG crops per agent inline as base64 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/22b2067b8f680812624032025864c5a1-Paper-Datasets_and_Benchmarks_Track.pdf) | +15–25% accuracy; saturates at 3 examples (1 clear positive, 1 partial/occluded, 1 low-light) | All 5 | Not implemented | |
| G1 | **Confidence-tiered Phase 2 cascade** — ≥0.88: auto-confirm skip Phase 2; 0.60–0.88: gemini-2.5-flash; <0.60: gemini-2.5-pro or discard | [arXiv 2601.06204](https://arxiv.org/html/2601.06204); [ViTA CVPR 2024] | 40–60% cost reduction on Gemini Pro calls; no meaningful accuracy drop | All 5 | Not implemented | |
| G2 | **JSON schema enforcement** — pass `response_format={"type":"json_schema",...}` to GPT; `response_schema` to Gemini | OpenAI + Google API docs | Eliminates parse failures; ~5% retry token reduction | All 5 | Not implemented | |
| G3 | **Camera ROI caching** — cache confirmed ROI per camera-angle pattern in `roi_cache.json`; skip Gemini ROI call on cache hit | Internal analysis | Eliminates one Gemini 2.5 Pro call per repeated job | All 5 | Not implemented | |
| G4 | **Static-patch deduplication** — perceptual hash of 16×16 ROI tiles; skip encoding tiles identical to previous frame | [EVS arXiv 2510.14624] | 3–4× token reduction (60–80% static regions in kitchen CCTV) | All 5 | Not implemented | |
| G5 | **Motion-based chunk pre-screening** — mean absolute frame-diff within ROI per chunk; skip if below threshold | [DyToK arXiv 2512.06866] | 30–50% fewer Phase 1 API calls on idle periods | All 5 | Not implemented | |

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
