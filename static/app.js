/**
 * Pork Weighing Analysis — Client-side Application Logic
 */

// =============================================================================
// Firebase Init
// =============================================================================
const firebaseConfig = {
    apiKey: "AIzaSyBq5vcef2PxzLKkW8ZCuloreuqhGa55QCA",
    authDomain: "kitchen-analysis-dashboard.firebaseapp.com",
    projectId: "kitchen-analysis-dashboard",
    storageBucket: "kitchen-analysis-dashboard.firebasestorage.app",
    messagingSenderId: "945227384148",
    appId: "1:945227384148:web:4d7f73ad31390246ff74c1",
    measurementId: "G-E54CNHNG2T"
};

// Initialize Firebase only if the script is loaded
if (typeof firebase !== 'undefined') {
    firebase.initializeApp(firebaseConfig);
}

// =============================================================================
// State
// =============================================================================
const state = {
    selectedFile: null,
    videoPath: '',
    serverFilename: null,
    roiCoords: null,             // { x1, y1, x2, y2 } — scale region
    timestampCoords: null,       // { x1, y1, x2, y2 } — timestamp region
    timestampFrameB64: null,     // cached frame for timestamp canvas
    recordingDate: null,
    recordingHour: null,
    pollingInterval: null,
    activeJobId: null,
    jobs: [],
    pendingDeletes: new Set(),
    activeCharts: {},
    selectedAgent: 'pork_weighing',
};

const $ = (sel) => document.querySelector(sel);

// =============================================================================
// Init
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initAuth();
});

// =============================================================================
// Auth
// =============================================================================
function initAuth() {
    if (typeof firebase === 'undefined') {
        // Fallback for local testing if Firebase scripts aren't loaded
        $('#auth-overlay').classList.remove('active');
        initApp();
        return;
    }

    firebase.auth().onAuthStateChanged((user) => {
        if (user) {
            // User is signed in.
            $('#auth-overlay').classList.remove('active');
            // If the app isn't initialized yet, initialize it
            if (!state.appInitialized) {
                state.appInitialized = true;
                initApp();
            }
        } else {
            // No user is signed in.
            $('#auth-overlay').classList.add('active');
            state.appInitialized = false;
        }
    });

    $('#login-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const email = $('#login-email').value;
        const password = $('#login-password').value;
        const btn = $('#login-btn');
        const errDiv = $('#login-error');

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Signing in...';
        errDiv.style.display = 'none';

        firebase.auth().signInWithEmailAndPassword(email, password)
            .catch((error) => {
                errDiv.textContent = error.message;
                errDiv.style.display = 'block';
                btn.disabled = false;
                btn.innerHTML = 'Sign In';
            });
    });
}

function initApp() {
    initUploadForm();
    initAgentTabs();
    initTimestampModal();
    loadJobs();
    startPolling();
}

// =============================================================================
// Toast
// =============================================================================
function showToast(message, type = 'info', duration = 4000) {
    const container = $('#toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// =============================================================================
// Tabs
// =============================================================================
function updateTaskUI(tabEl) {
    const roiLabel = tabEl.dataset.roiLabel || 'Detect Scale ROI';
    const phase2Label = tabEl.dataset.phase2Label || 'Phase 2 Verification';
    const taskDesc = tabEl.dataset.taskDesc || '';

    const roiBtn = document.getElementById('auto-detect-roi-btn');
    if (roiBtn) roiBtn.textContent = `🔍 Step 2: ${roiLabel}`;

    const p2Label = document.getElementById('phase2-toggle-label');
    if (p2Label) p2Label.textContent = phase2Label;

    const descEl = document.getElementById('task-description');
    if (descEl) descEl.textContent = taskDesc;
}

function handleFormResize() {
    const panels = document.querySelectorAll('.jobs-section, .results-panel');
    panels.forEach(p => {
        p.style.opacity = '1';
    });
}

// ── Image Lightbox Modal Functions ──────────────────────────────────────────
window.openImageModal = function (src) {
    const modal = document.getElementById('image-modal-overlay');
    const img = document.getElementById('lightbox-img');
    if (modal && img) {
        img.src = src;
        modal.classList.add('active');
    }
};

window.closeImageModal = function () {
    const modal = document.getElementById('image-modal-overlay');
    if (modal) {
        modal.classList.remove('active');
        // Clear src after fade out to avoid flash
        setTimeout(() => {
            const img = document.getElementById('lightbox-img');
            if (img) img.src = '';
        }, 300);
    }
};

function initAgentTabs() {
    const tabs = document.querySelectorAll('.agent-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            tabs.forEach(t => {
                t.classList.remove('btn-primary', 'active');
                t.classList.add('btn-secondary');
            });
            const clicked = e.target.closest('.agent-tab');
            clicked.classList.remove('btn-secondary');
            clicked.classList.add('btn-primary', 'active');
            state.selectedAgent = clicked.dataset.agent;

            updateTaskUI(clicked);

            // Show toast indicating agent change
            showToast(`Switched to ${clicked.textContent.trim()} agent`, 'info');
        });
    });

    // Initialise UI for the default active tab
    const activeTab = document.querySelector('.agent-tab.active');
    if (activeTab) updateTaskUI(activeTab);
}

// =============================================================================
// Upload Form & Drag-Drop
// =============================================================================
function initUploadForm() {
    const fileInput = $('#video-file-input');
    const pathInput = $('#video-path-input');
    const submitBtn = $('#submit-job-btn');
    const tsBtn = $('#select-timestamp-btn');

    fileInput?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const maxSize = parseInt(document.body.dataset.maxUploadMb || 500);
        if (file.size > maxSize * 1024 * 1024) {
            showToast(`File exceeds ${maxSize}MB limit. Use file path instead.`, 'error');
            fileInput.value = '';
            return;
        }
        state.selectedFile = file;
        state.videoPath = '';
        if (pathInput) pathInput.value = '';
        updateFileDisplay(file);
        resetTimestampState();
        resetROIState();
        preUploadAndDetect(file);
    });

    pathInput?.addEventListener('input', (e) => {
        state.videoPath = e.target.value.trim();
        if (state.videoPath) {
            state.selectedFile = null;
            if (fileInput) fileInput.value = '';
            hideFileDisplay();
            resetTimestampState();
            resetROIState();
            setTimeout(() => Promise.allSettled([autoDetectTimestamp(), autoDetectROI()]), 1000);
        }
    });

    submitBtn?.addEventListener('click', submitJob);
    tsBtn?.addEventListener('click', openTimestampSelector);
    $('#auto-detect-roi-btn')?.addEventListener('click', autoDetectROI);

    // Drag & Drop
    const dropzone = $('#upload-dropzone');
    if (!dropzone) return;
    ['dragenter', 'dragover'].forEach(evt =>
        dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(evt =>
        dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); }));
    dropzone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (!file) return;
        const maxSize = parseInt(document.body.dataset.maxUploadMb || 500);
        if (file.size > maxSize * 1024 * 1024) {
            showToast(`File exceeds ${maxSize}MB limit.`, 'error');
            return;
        }
        state.selectedFile = file;
        state.videoPath = '';
        const pi = $('#video-path-input');
        if (pi) pi.value = '';
        updateFileDisplay(file);
        const fi = $('#video-file-input');
        if (fi) { const dt = new DataTransfer(); dt.items.add(file); fi.files = dt.files; }
        resetTimestampState();
        resetROIState();
        preUploadAndDetect(file);
    });
}

function updateFileDisplay(file) {
    const d = $('#selected-file-display');
    if (!d) return;
    d.classList.add('visible');
    d.querySelector('.file-name').textContent = file.name;
    d.querySelector('.file-size').textContent = formatBytes(file.size);
}
function hideFileDisplay() {
    const d = $('#selected-file-display');
    if (d) d.classList.remove('visible');
}
function formatBytes(bytes) {
    if (!bytes) return '0 B';
    const k = 1024, sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// =============================================================================
// Shared: fetch a frame from the current video
// =============================================================================
async function fetchVideoFrame(frameIndex = 30) {
    if (state.selectedFile) {
        // Extract frame locally in browser to save bandwidth
        return new Promise((resolve, reject) => {
            const fileURL = URL.createObjectURL(state.selectedFile);
            const video = document.createElement('video');
            video.preload = 'metadata';
            video.src = fileURL;
            video.muted = true;
            video.playsInline = true;

            video.onloadedmetadata = () => {
                // Approximate time: frameIndex / 30fps
                const targetTime = Math.min(frameIndex / 30, video.duration || 1);
                video.currentTime = targetTime;
            };

            video.onseeked = () => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                // Get base64 without prefix
                const dataURL = canvas.toDataURL('image/jpeg', 0.95);
                const b64 = dataURL.split(',')[1];
                URL.revokeObjectURL(fileURL);
                resolve(b64);
            };

            video.onerror = () => {
                URL.revokeObjectURL(fileURL);
                reject(new Error('Failed to load video locally for frame extraction.'));
            };
        });
    }

    // Fallback for file paths
    const formData = new FormData();
    formData.append('video_path', state.videoPath);
    formData.append('frame_index', frameIndex);

    const resp = await fetch('/api/extract-frame', { method: 'POST', body: formData });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Frame extraction failed'); }
    return (await resp.json()).frame;
}

// =============================================================================
// Generic canvas region selector (used by timestamp modal)
// =============================================================================
function makeCanvasSelector({ canvasId, coordsDisplayId }) {
    let canvas, ctx, image;
    let drawing = false, sx = 0, sy = 0, cx = 0, cy = 0;
    let scaleX = 1, scaleY = 1;

    function load(frameB64) {
        canvas = document.getElementById(canvasId);
        ctx = canvas.getContext('2d');
        image = new Image();
        image.onload = () => {
            const maxW = 850, maxH = 500;
            let w = image.width, h = image.height;
            if (w > maxW) { h = h * maxW / w; w = maxW; }
            if (h > maxH) { w = w * maxH / h; h = maxH; }
            canvas.width = w; canvas.height = h;
            scaleX = image.width / w; scaleY = image.height / h;
            ctx.drawImage(image, 0, 0, w, h);
        };
        image.src = `data:image/jpeg;base64,${frameB64}`;

        canvas.onmousedown = (e) => {
            const r = canvas.getBoundingClientRect();
            sx = Math.max(0, Math.min(e.clientX - r.left, canvas.width));
            sy = Math.max(0, Math.min(e.clientY - r.top, canvas.height));
            drawing = true;
        };
        canvas.onmousemove = (e) => {
            if (!drawing) return;
            const r = canvas.getBoundingClientRect();
            cx = Math.max(0, Math.min(e.clientX - r.left, canvas.width));
            cy = Math.max(0, Math.min(e.clientY - r.top, canvas.height));
            redraw();
        };
        canvas.onmouseup = (e) => {
            if (!drawing) return;
            drawing = false;
            const r = canvas.getBoundingClientRect();
            cx = Math.max(0, Math.min(e.clientX - r.left, canvas.width));
            cy = Math.max(0, Math.min(e.clientY - r.top, canvas.height));
            updateDisplay();
        };
        canvas.onmouseleave = (e) => {
            if (drawing) {
                drawing = false;
                const r = canvas.getBoundingClientRect();
                cx = Math.max(0, Math.min(e.clientX - r.left, canvas.width));
                cy = Math.max(0, Math.min(e.clientY - r.top, canvas.height));
                updateDisplay();
            }
        };
    }

    function redraw() {
        if (!image) return;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        drawRect(sx, sy, cx, cy);
        updateDisplay();
    }

    function drawRect(x1, y1, x2, y2) {
        ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]); ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.fillRect(0, 0, canvas.width, Math.min(y1, y2));
        ctx.fillRect(0, Math.max(y1, y2), canvas.width, canvas.height - Math.max(y1, y2));
        ctx.fillRect(0, Math.min(y1, y2), Math.min(x1, x2), Math.abs(y2 - y1));
        ctx.fillRect(Math.max(x1, x2), Math.min(y1, y2), canvas.width - Math.max(x1, x2), Math.abs(y2 - y1));
    }

    function updateDisplay() {
        const el = document.getElementById(coordsDisplayId);
        if (!el) return;
        const rx1 = Math.round(Math.min(sx, cx) * scaleX);
        const ry1 = Math.round(Math.min(sy, cy) * scaleY);
        const rx2 = Math.round(Math.max(sx, cx) * scaleX);
        const ry2 = Math.round(Math.max(sy, cy) * scaleY);
        el.textContent = `Region: (${rx1}, ${ry1}) → (${rx2}, ${ry2})  |  ${rx2 - rx1} × ${ry2 - ry1} px`;
    }

    function getCoords() {
        return {
            x1: Math.round(Math.min(sx, cx) * scaleX),
            y1: Math.round(Math.min(sy, cy) * scaleY),
            x2: Math.round(Math.max(sx, cx) * scaleX),
            y2: Math.round(Math.max(sy, cy) * scaleY),
        };
    }

    function reset() {
        sx = sy = cx = cy = 0;
        if (ctx && image) ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        const el = document.getElementById(coordsDisplayId);
        if (el) el.textContent = 'No region selected';
    }

    function isValid() {
        const c = getCoords();
        return (c.x2 - c.x1) > 2 && (c.y2 - c.y1) > 2;
    }

    return { load, reset, getCoords, isValid };
}

// =============================================================================
// Pre-upload: upload file once, then trigger ROI + timestamp detection
// =============================================================================

function _setUploadProgress(pct, statusText) {
    const bar = $('#upload-progress-bar');
    const wrap = $('#upload-progress-wrap');
    const txt = $('#upload-status-text');
    if (wrap) wrap.style.display = pct >= 100 ? 'none' : 'block';
    if (bar) bar.style.width = `${pct}%`;
    if (txt) { txt.textContent = statusText || ''; txt.style.display = statusText ? 'inline' : 'none'; }
}

function preUploadAndDetect(file) {
    // Disable ROI button and show uploading state
    const roiBtn = $('#auto-detect-roi-btn');
    if (roiBtn) { roiBtn.disabled = true; roiBtn.innerHTML = '<span class="spinner"></span> Uploading...'; }

    _setUploadProgress(0, 'Uploading to server…');

    const fd = new FormData();
    fd.append('video_file', file);

    const xhr = new XMLHttpRequest();

    // Live upload progress
    xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const pct = Math.round((e.loaded / e.total) * 100);
        _setUploadProgress(pct, `Uploading… ${pct}%`);
        if (roiBtn) roiBtn.innerHTML = `<span class="spinner"></span> Uploading ${pct}%`;
    };

    xhr.onload = () => {
        _setUploadProgress(100, '');

        if (xhr.status < 200 || xhr.status >= 300) {
            let detail = 'Upload failed';
            try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) { }
            showToast(`Upload failed: ${detail}`, 'error');
            if (roiBtn) { roiBtn.disabled = false; roiBtn.innerHTML = '🔍 Step 2: Detect ROI'; }
            return;
        }

        let data;
        try { data = JSON.parse(xhr.responseText); } catch (_) {
            showToast('Upload failed: invalid server response', 'error');
            if (roiBtn) { roiBtn.disabled = false; roiBtn.innerHTML = '🔍 Step 2: Detect ROI'; }
            return;
        }

        // Switch to server-path mode
        state.videoPath = data.path;
        state.serverFilename = data.filename;
        state.selectedFile = null;

        if (roiBtn) roiBtn.innerHTML = '<span class="spinner"></span> Detecting ROI…';
        _setUploadProgress(100, 'Detecting ROI…');

        // Trigger both detections in parallel
        Promise.allSettled([autoDetectTimestamp(), autoDetectROI()]).then(() => {
            _setUploadProgress(100, '');
        });
    };

    xhr.onerror = () => {
        _setUploadProgress(100, '');
        showToast('Upload failed: network error', 'error');
        if (roiBtn) { roiBtn.disabled = false; roiBtn.innerHTML = '🔍 Step 2: Detect ROI'; }
    };

    xhr.open('POST', '/api/upload-video');
    xhr.send(fd);
}


// =============================================================================
// STEP 1: Auto Timestamp Detection
// =============================================================================

async function autoDetectTimestamp() {
    if (!state.selectedFile && !state.videoPath) return;

    const badge = $('#timestamp-display');
    if (badge) {
        badge.style.display = 'inline-flex';
        badge.textContent = 'Detecting timestamp...';
    }

    const tsStatus = $('#ts-auto-status');
    if (tsStatus) {
        tsStatus.style.display = 'block';
        tsStatus.textContent = 'Detecting...';
        tsStatus.className = 'ts-auto-status detecting';
    }

    try {
        const formData = new FormData();
        if (state.selectedFile) {
            formData.append('video_file', state.selectedFile);
        } else {
            formData.append('video_path', state.videoPath);
        }
        const rotInput = $('#adv-rotation');
        formData.append('rotation_angle', rotInput?.value ? parseInt(rotInput.value) : 0);

        const resp = await fetch('/api/auto-detect-timestamp', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Timestamp detection request failed');

        const data = await resp.json();

        if (data.recording_date && data.recording_hour !== null) {
            state.recordingDate = data.recording_date;
            state.recordingHour = data.recording_hour;

            if (badge) {
                badge.style.display = 'inline-flex';
                badge.textContent = `✓ ${data.recording_date} ${String(data.recording_hour).padStart(2, '0')}:xx`;
            }

            // Pre-fill manual override fields in the modal
            const md = $('#manual-date'); if (md) md.value = data.recording_date;
            const mh = $('#manual-hour'); if (mh) mh.value = data.recording_hour;

            if (tsStatus) {
                tsStatus.textContent = `Detected: ${data.raw_timestamp || `${data.recording_date} ${data.recording_hour}:xx`}`;
                tsStatus.className = 'ts-auto-status success';
            }
        } else {
            if (badge) badge.style.display = 'none';
            if (tsStatus) {
                tsStatus.textContent = 'Not detected — use Step 1 to set manually';
                tsStatus.className = 'ts-auto-status failed';
            }
        }
    } catch (err) {
        if (badge) badge.style.display = 'none';
        if (tsStatus) {
            tsStatus.textContent = 'Detection failed — use Step 1 to set manually';
            tsStatus.className = 'ts-auto-status failed';
        }
        logger.debug?.(`Timestamp auto-detect error: ${err.message}`);
    }
}

// =============================================================================
// STEP 1: Timestamp Region Modal (manual override)
// =============================================================================
let tsSelector = null;

function initTimestampModal() {
    tsSelector = makeCanvasSelector({
        canvasId: 'timestamp-canvas',
        coordsDisplayId: 'timestamp-coords-text',
    });

    $('#timestamp-modal-close')?.addEventListener('click', closeTimestampModal);
    $('#timestamp-cancel-btn')?.addEventListener('click', closeTimestampModal);
    $('#timestamp-reset-btn')?.addEventListener('click', () => {
        tsSelector.reset();
        const r = $('#timestamp-ocr-result');
        if (r) r.style.display = 'none';
    });
    $('#timestamp-run-ocr-btn')?.addEventListener('click', runTimestampOCR);
    $('#timestamp-confirm-btn')?.addEventListener('click', confirmTimestamp);
}

async function openTimestampSelector() {
    if (!state.selectedFile && !state.videoPath) {
        showToast('Upload a video or enter a file path first.', 'error');
        return;
    }
    const overlay = $('#timestamp-modal-overlay');
    overlay.classList.add('active');

    // Hide OCR result until run
    const r = $('#timestamp-ocr-result');
    if (r) r.style.display = 'none';

    showToast('Loading frame...', 'info', 1500);
    try {
        // Use frame 30 (same as ROI step) to ensure consistency and avoid intro/black frames
        const frameB64 = await fetchVideoFrame(30);
        state.timestampFrameB64 = frameB64;
        tsSelector.load(frameB64);
    } catch (err) {
        showToast(`Frame error: ${err.message}`, 'error');
    }
}

async function runTimestampOCR() {
    if (!tsSelector.isValid()) {
        showToast('Draw a rectangle around the timestamp area first.', 'error');
        return;
    }
    if (!state.timestampFrameB64) {
        showToast('No frame loaded.', 'error');
        return;
    }

    const btn = $('#timestamp-run-ocr-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running OCR...';

    try {
        const coords = tsSelector.getCoords();
        const resp = await fetch('/api/extract-timestamp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame_b64: state.timestampFrameB64,
                region: [coords.x1, coords.y1, coords.x2, coords.y2],
            }),
        });
        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'OCR failed'); }

        const data = await resp.json();
        const resultDiv = $('#timestamp-ocr-result');
        resultDiv.style.display = 'block';
        $('#ts-date-val').textContent = data.recording_date || '—';
        $('#ts-hour-val').textContent = data.recording_hour !== null ? data.recording_hour : '—';
        $('#ts-raw-val').textContent = data.raw_timestamp || '—';
        if (data.recording_date) $('#manual-date').value = data.recording_date;
        if (data.recording_hour !== null) $('#manual-hour').value = data.recording_hour;
        showToast('Timestamp OCR complete!', 'success');
    } catch (err) {
        showToast(`OCR Error: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🔍 Run OCR';
    }
}

function confirmTimestamp() {
    if (!tsSelector.isValid()) {
        showToast('Draw a rectangle around the timestamp area first.', 'error');
        return;
    }

    // Use manual override if filled, else OCR result
    const manualDate = $('#manual-date')?.value.trim();
    const manualHour = $('#manual-hour')?.value;
    const dateVal = manualDate || $('#ts-date-val')?.textContent;
    const hourStr = manualHour !== '' ? manualHour : $('#ts-hour-val')?.textContent;
    const hourVal = parseInt(hourStr);

    if (!dateVal || dateVal === '—') {
        showToast('Run OCR first or enter date manually.', 'error');
        return;
    }
    if (isNaN(hourVal) || hourVal < 0 || hourVal > 23) {
        showToast('Invalid hour. Enter 0–23.', 'error');
        return;
    }

    state.timestampCoords = tsSelector.getCoords();
    state.recordingDate = dateVal;
    state.recordingHour = hourVal;

    const badge = $('#timestamp-display');
    if (badge) {
        badge.style.display = 'inline-flex';
        badge.textContent = `✓ ${dateVal} ${String(hourVal).padStart(2, '0')}:xx`;
    }

    closeTimestampModal();
    showToast(`Timestamp set: ${dateVal} ${String(hourVal).padStart(2, '0')}:00`, 'success');
}

function closeTimestampModal() {
    $('#timestamp-modal-overlay').classList.remove('active');
}

function resetTimestampState() {
    state.timestampCoords = null;
    state.recordingDate = null;
    state.recordingHour = null;
    state.timestampFrameB64 = null;
    const badge = $('#timestamp-display');
    if (badge) badge.style.display = 'none';
    const r = $('#timestamp-ocr-result');
    if (r) r.style.display = 'none';
    const md = $('#manual-date'); if (md) md.value = '';
    const mh = $('#manual-hour'); if (mh) mh.value = '';
    const tsStatus = $('#ts-auto-status');
    if (tsStatus) { tsStatus.style.display = 'none'; tsStatus.textContent = ''; }
}

// =============================================================================
// STEP 2: Auto ROI Detection
// =============================================================================

function resetROIState() {
    state.roiCoords = null;
    const badge = $('#roi-display');
    if (badge) badge.style.display = 'none';
    const preview = $('#roi-auto-preview');
    if (preview) {
        preview.style.display = 'none';
        const img = preview.querySelector('#roi-preview-img');
        if (img) img.src = '';
    }
}

async function autoDetectROI() {
    if (!state.selectedFile && !state.videoPath) {
        showToast('Upload a video or enter a file path first.', 'error');
        return;
    }

    const btn = $('#auto-detect-roi-btn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Detecting ROI...';
    }

    // Show preview area in "detecting" state
    const preview = $('#roi-auto-preview');
    if (preview) {
        preview.style.display = 'flex';
        const statusEl = preview.querySelector('#roi-detect-status');
        if (statusEl) statusEl.textContent = 'Detecting ROI...';
        const badgeEl = preview.querySelector('#roi-method-badge');
        if (badgeEl) { badgeEl.textContent = 'Detecting...'; badgeEl.className = 'roi-method-badge'; }
    }

    try {
        const formData = new FormData();
        if (state.selectedFile) {
            formData.append('video_file', state.selectedFile);
        } else {
            formData.append('video_path', state.videoPath);
        }
        // Pass current rotation if set in advanced config
        const rotInput = $('#adv-rotation');
        const rotation = rotInput?.value ? parseInt(rotInput.value) : 0;
        formData.append('rotation_angle', rotation);
        // Pass the active agent so the VLM prompt is task-specific
        formData.append('agent', state.selectedAgent || 'pork_weighing');

        const resp = await fetch('/api/auto-detect-roi', { method: 'POST', body: formData });
        if (!resp.ok) {
            const e = await resp.json();
            throw new Error(e.detail || 'ROI detection request failed');
        }

        const data = await resp.json();

        if (!data.roi) {
            _showROIFailedState();
            showToast('Could not detect scale ROI automatically. Job will run without ROI crop.', 'error');
            return;
        }

        const [x1, y1, x2, y2] = data.roi;
        state.roiCoords = { x1, y1, x2, y2 };

        const badge = $('#roi-display');
        if (badge) { badge.style.display = 'inline-flex'; badge.textContent = 'ROI Detected'; }

        _showROIPreview(data);

        const methodLabel = data.method === 'vlm' ? 'AI Vision' : 'Auto-Detect';
        const pct = Math.round(data.confidence * 100);
        showToast(`ROI detected via ${methodLabel} (${pct}% confidence)`, 'success');

    } catch (err) {
        showToast(`ROI Detection Error: ${err.message}`, 'error');
        _showROIFailedState();
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '🔍 Re-detect ROI';
        }
    }
}

function _showROIPreview(data) {
    const preview = $('#roi-auto-preview');
    if (!preview) return;
    preview.style.display = 'flex';

    const img = preview.querySelector('#roi-preview-img');
    if (img && data.annotated_frame) {
        img.src = `data:image/jpeg;base64,${data.annotated_frame}`;
    }

    const methodBadge = preview.querySelector('#roi-method-badge');
    if (methodBadge) {
        const methodLabel = data.method === 'vlm' ? 'AI Vision' : 'Auto-Detect';
        const pct = Math.round(data.confidence * 100);
        methodBadge.textContent = `${methodLabel} · ${pct}%`;
        methodBadge.className = 'roi-method-badge ' + (
            data.confidence >= 0.7 ? 'confidence-high' :
                data.confidence >= 0.4 ? 'confidence-mid' : 'confidence-low'
        );
    }

    const statusEl = preview.querySelector('#roi-detect-status');
    if (statusEl && data.roi) {
        const [x1, y1, x2, y2] = data.roi;
        statusEl.textContent = `(${x1}, ${y1}) → (${x2}, ${y2})`;
    }
}

function _showROIFailedState() {
    const preview = $('#roi-auto-preview');
    if (!preview) return;
    preview.style.display = 'flex';

    const statusEl = preview.querySelector('#roi-detect-status');
    if (statusEl) statusEl.textContent = 'Detection failed — job will run without ROI crop';

    const methodBadge = preview.querySelector('#roi-method-badge');
    if (methodBadge) {
        methodBadge.textContent = 'Failed';
        methodBadge.className = 'roi-method-badge confidence-low';
    }

}

// =============================================================================
// Job Submission
// =============================================================================
// Stop Server
// =============================================================================
async function stopServer() {
    const btn = $('#stop-server-btn');
    if (!confirm('Stop the server? You will need to restart it manually from the terminal.')) return;

    btn.disabled = true;
    btn.textContent = '⏹ Stopping…';

    try {
        await fetch('/api/stop', { method: 'POST' });
    } catch (_) {
        // Expected: server closes the connection as it shuts down
    }

    document.body.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column;gap:16px;font-family:sans-serif;">
            <div style="font-size:3rem;">⏹</div>
            <h2 style="margin:0;">Server stopped</h2>
            <p style="color:#888;margin:0;">Start the server again from the terminal, then refresh this page.</p>
        </div>`;
}

// =============================================================================
async function submitJob() {
    const submitBtn = $('#submit-job-btn');
    if (!state.selectedFile && !state.videoPath) {
        showToast('Upload a video or enter a file path.', 'error');
        return;
    }

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Submitting...';

    try {
        const formData = new FormData();

        if (state.selectedFile) {
            formData.append('video_file', state.selectedFile);
        } else {
            formData.append('video_path', state.videoPath);
            if (state.serverFilename) formData.append('original_filename', state.serverFilename);
        }

        if (state.roiCoords) {
            formData.append('roi_x1', state.roiCoords.x1);
            formData.append('roi_y1', state.roiCoords.y1);
            formData.append('roi_x2', state.roiCoords.x2);
            formData.append('roi_y2', state.roiCoords.y2);
        }
        if (state.timestampCoords) {
            formData.append('ts_x1', state.timestampCoords.x1);
            formData.append('ts_y1', state.timestampCoords.y1);
            formData.append('ts_x2', state.timestampCoords.x2);
            formData.append('ts_y2', state.timestampCoords.y2);
        }
        if (state.recordingHour !== null) formData.append('recording_hour', state.recordingHour);
        if (state.recordingDate) formData.append('recording_date', state.recordingDate);
        formData.append('agent', state.selectedAgent);

        const phase2Toggle = $('#enable-phase2');
        formData.append('enable_phase2', phase2Toggle ? phase2Toggle.checked : true);
        const fpsInput = $('#fps-input');
        if (fpsInput?.value) formData.append('fps', parseFloat(fpsInput.value));

        // Advanced config collection
        const advancedInputs = document.querySelectorAll('.adv-config-input');
        let advConfig = {};
        advancedInputs.forEach(input => {
            if (input.value !== '') {
                const isFloat = input.step && input.step.includes('.');
                advConfig[input.dataset.key] = isFloat ? parseFloat(input.value) : parseInt(input.value);
            }
        });
        if (Object.keys(advConfig).length > 0) {
            formData.append('advanced_config', JSON.stringify(advConfig));
        }

        submitBtn.innerHTML = '<span class="spinner"></span> Starting job...';
        const resp = await fetch('/api/jobs', { method: 'POST', body: formData });
        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Failed to create job'); }

        const data = await resp.json();
        showToast(`Job #${data.id} created! Processing will begin shortly.`, 'success');

        // Reset form
        state.selectedFile = null; state.videoPath = '';
        const fi = $('#video-file-input'); if (fi) fi.value = '';
        const pi = $('#video-path-input'); if (pi) pi.value = '';
        hideFileDisplay();
        resetTimestampState();
        resetROIState();

        await loadJobs();
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '🚀 Start Analysis';
    }
}

// =============================================================================
// Job Polling & Display
// =============================================================================
const POLL_INTERVAL_BASE = 10000;   // 10 s normal cadence
const POLL_INTERVAL_MAX = 120000;  // 2 min ceiling during quota backoff
let _pollBackoffMs = POLL_INTERVAL_BASE;

function startPolling() {
    // Use a self-rescheduling timeout so backoff can vary the interval
    _schedulePoll();
}

function _schedulePoll() {
    if (state.pollingInterval) clearTimeout(state.pollingInterval);
    state.pollingInterval = setTimeout(async () => {
        await loadJobs();
        _schedulePoll();
    }, _pollBackoffMs);
}

async function loadJobs() {
    try {
        const resp = await fetch('/api/jobs');
        if (resp.status === 429) {
            // Quota exceeded — double backoff, cap at max, show warning once
            _pollBackoffMs = Math.min(_pollBackoffMs * 2, POLL_INTERVAL_MAX);
            console.warn(`Quota exceeded — backing off polling to ${_pollBackoffMs / 1000}s`);
            return;
        }
        if (!resp.ok) return;
        // Successful fetch — reset backoff
        _pollBackoffMs = POLL_INTERVAL_BASE;
        const allJobs = await resp.json();
        // Filter out any jobs whose DELETE is still in-flight to prevent
        // the polling interval from resurrecting them before the server responds.
        state.jobs = allJobs.filter(j => !state.pendingDeletes.has(String(j.id)));
        renderJobsTable();
        if (state.activeJobId) {
            const active = state.jobs.find(j => j.id === state.activeJobId);
            if (active && ['queued', 'processing', 'phase1', 'phase2'].includes(active.status)) {
                viewJobResults(state.activeJobId);
            }
        }
    } catch (err) {
        console.error('Failed to load jobs:', err);
    }
}

function renderJobsTable() {
    const tbody = $('#jobs-tbody');
    const empty = $('#jobs-empty');
    if (!tbody) return;
    if (state.jobs.length === 0) {
        tbody.innerHTML = '';
        if (empty) empty.style.display = 'block';
        return;
    }
    if (empty) empty.style.display = 'none';
    tbody.innerHTML = state.jobs.map(job => `
        <tr class="clickable" onclick="viewJobResults('${job.id}')">
            <td><strong>#<span style="font-size:0.7em">${job.id.substring(0, 8)}</span></strong></td>
            <td class="video-name-cell" title="${escapeHtml(job.video_name)}">
                <div style="font-weight: 600; font-size: 0.8em; color: var(--text-muted);">${job.agent || 'Pork'}</div>
                ${escapeHtml(job.video_name)}
            </td>
            <td><span class="status-badge status-${job.status}">${getStatusIcon(job.status)} ${job.status}</span></td>
            <td class="progress-msg">
                <div style="margin-bottom: 4px;">${escapeHtml(job.progress_message || '—')}</div>
                ${renderProgressBar(job)}
            </td>
            <td class="time-cell">${formatTime(job.created_at)}</td>
            <td class="actions-cell" onclick="event.stopPropagation()">
                <button class="btn btn-danger btn-sm" onclick="deleteJob('${job.id}')" title="Delete">🗑️</button>
            </td>
        </tr>
    `).join('');
}

function renderProgressBar(job) {
    if (job.status === 'completed') return '<div style="height:4px;width:100%;background:var(--accent-green);border-radius:2px;"></div>';
    if (job.status === 'failed') return '<div style="height:4px;width:100%;background:var(--accent-red);border-radius:2px;"></div>';
    if (job.status === 'queued') return '<div style="height:4px;width:100%;background:rgba(255,255,255,0.1);border-radius:2px;"></div>';

    let pct = 0;
    if (job.status === 'processing') pct = 10;

    // Parse progress from message e.g. "Analyzing batch 2/10" or "Phase 1: 50%"
    let msg = job.progress_message || '';
    let match = msg.match(/[bB]atch\s+(\d+)\/(\d+)/) || msg.match(/[cC]lip\s+(\d+)\/(\d+)/);

    if (match) {
        let current = parseInt(match[1]);
        let total = parseInt(match[2]);
        if (total > 0) {
            let phasePct = (current / total) * 100;
            if (msg.toLowerCase().includes('phase 1') || job.status === 'phase1') {
                pct = 10 + (phasePct * 0.4); // Phase 1 takes 40% (10-50%)
            } else if (msg.toLowerCase().includes('phase 2') || job.status === 'phase2') {
                pct = 50 + (phasePct * 0.45); // Phase 2 takes 45% (50-95%)
            } else {
                pct = phasePct;
            }
        }
    } else {
        if (msg.toLowerCase().includes('ocr') || msg.toLowerCase().includes('timestamp')) pct = 5;
        else if (job.status === 'phase1') pct = 20 + Math.random() * 20; // fallback animated indeterminate
        else if (job.status === 'phase2') pct = 60 + Math.random() * 20;
    }

    return `<div style="height:6px;width:100%;background:rgba(255,255,255,0.1);border-radius:3px;overflow:hidden;margin-top:2px;">
                <div style="height:100%;width:${pct}%;background:var(--accent-cyan);transition:width 0.5s ease;border-radius:3px;${(!match && ['processing', 'phase1', 'phase2'].includes(job.status)) ? 'animation: pulse 1s infinite alternate;' : ''}"></div>
            </div>`;
}

function getStatusIcon(status) {
    return { queued: '🔵', processing: '⚙️', phase1: '🔍', phase2: '✅', completed: '✨', failed: '❌' }[status] || '❓';
}
function formatTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
// Convert total seconds to HH:MM:SS string
function secondsToHms(totalSec) {
    const s = Math.round(totalSec);
    const hh = Math.floor(s / 3600);
    const mm = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}
// Add offset seconds to an HH:MM:SS string and return a new HH:MM:SS string
function addSecondsToHms(hms, offsetSec) {
    const [h, m, s] = hms.split(':').map(Number);
    return secondsToHms(h * 3600 + m * 60 + s + offsetSec);
}

// Compute real-world HH:MM:SS based on video seconds and OCR recording hour
function computeRealHms(seconds, startHour) {
    const s = Math.round(seconds);
    const h = Math.floor(s / 3600) + (startHour || 0);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}
function escapeHtml(str) {
    if (!str) return '';
    const d = document.createElement('div'); d.textContent = str; return d.innerHTML;
}

// =============================================================================
// View Results
// =============================================================================
async function viewJobResults(jobId) {
    state.activeJobId = jobId;
    try {
        const resp = await fetch(`/api/jobs/${jobId}`);
        if (!resp.ok) throw new Error('Failed to load job');
        const job = await resp.json();

        // Auto-load frames
        job.framesData = [];
        if (job.status === 'completed' || job.status === 'phase2') {
            try {
                const fResp = await fetch(`/api/jobs/${jobId}/frames`);
                if (fResp.ok) {
                    const fData = await fResp.json();
                    job.framesData = fData.frames || [];
                }
            } catch (e) {
                console.warn('Could not load frames', e);
            }
        }

        renderResultsPanel(job);
    } catch (err) {
        showToast(`Error loading results: ${err.message}`, 'error');
    }
}

async function renderResultsPanel(job) {
    const panel = $('#results-panel');
    if (!panel) return;
    panel.classList.add('active');

    const results = job.results || [];
    const agent = job.agent || 'pork_weighing';
    const tokenSummary = job.token_summary || {};

    // ── Agent-specific KPI computation ──────────────────────────────────────
    let agentKpiHtml = '';
    if (agent === 'serve_time') {
        const times = results.map(r => r.verified_reading).filter(v => v != null && !isNaN(v));
        const avg = times.length ? times.reduce((a, b) => a + b, 0) / times.length : null;
        const fmtDur = s => s == null ? '—' : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
        agentKpiHtml = `
            <div class="stat-card"><div class="stat-value">${fmtDur(avg)}</div><div class="stat-label">Avg Serve Time</div></div>`;
    } else if (agent === 'plating_time') {
        const times = results.map(r => r.verified_reading).filter(v => v != null && !isNaN(v));
        const avg = times.length ? times.reduce((a, b) => a + b, 0) / times.length : null;
        const fmtSec = s => s == null ? '—' : `${s.toFixed(1)}s`;
        agentKpiHtml = `
            <div class="stat-card"><div class="stat-value">${fmtSec(avg)}</div><div class="stat-label">Avg Plating Time</div></div>
            <div class="stat-card"><div class="stat-value">${fmtSec(times.length ? Math.min(...times) : null)}</div><div class="stat-label">Min Plating Time</div></div>
            <div class="stat-card"><div class="stat-value">${fmtSec(times.length ? Math.max(...times) : null)}</div><div class="stat-label">Max Plating Time</div></div>`;
    } else if (agent === 'noodle_rotation') {
        const counts = results.map(r => r.verified_reading ?? r.scale).filter(v => v != null && !isNaN(Number(v)));
        const avgCount = counts.length ? (counts.reduce((a, b) => a + Number(b), 0) / counts.length).toFixed(1) : '—';
        const durations = results.map(r => (r.end_time || 0) - (r.start_time || 0)).filter(d => d > 0);
        const avgDur = durations.length ? (durations.reduce((a, b) => a + b, 0) / durations.length).toFixed(1) : '—';
        agentKpiHtml = `
            <div class="stat-card"><div class="stat-value">${results.length}</div><div class="stat-label">Total Events</div></div>
            <div class="stat-card"><div class="stat-value">${avgCount}</div><div class="stat-label">Avg Rotations / Event</div></div>
            <div class="stat-card"><div class="stat-value">${avgDur}s</div><div class="stat-label">Avg Event Duration</div></div>`;
    } else if (agent === 'bowl_completion_rate') {
        const completed = results.filter(r => r.scale === 'COMPLETED').length;
        const total = results.length;
        const rate = total > 0 ? ((completed / total) * 100).toFixed(1) : '—';
        agentKpiHtml = `
            <div class="stat-card"><div class="stat-value">${rate}%</div><div class="stat-label">Completion Rate</div></div>
            <div class="stat-card"><div class="stat-value">${completed}</div><div class="stat-label">Completed</div></div>
            <div class="stat-card"><div class="stat-value">${total - completed}</div><div class="stat-label">Not Completed</div></div>`;
    } else {
        // pork_weighing
        const weights = results.map(r => parseFloat(r.verified_reading ?? r.phase1_reading)).filter(v => !isNaN(v));
        const isStandard = w => (Math.abs(w - 60) <= 5) || (Math.abs(w - 120) <= 5);
        const standardCount = weights.filter(isStandard).length;
        const nonStandardCount = weights.length - standardCount;

        agentKpiHtml = `
            <div class="stat-card"><div class="stat-value">${standardCount}</div><div class="stat-label">Standard Portions</div></div>
            <div class="stat-card"><div class="stat-value" style="color:var(--accent-red);">${nonStandardCount}</div><div class="stat-label">Non-Standard Portions</div></div>
            <div class="stat-card"><div class="stat-value">${results.length}</div><div class="stat-label">Total Weighing Events</div></div>`;
    }

    // ── Agent-specific chart HTML ────────────────────────────────────────────
    let chartsHtml = '';
    if (agent === 'serve_time' && results.length > 0) {
        chartsHtml = `<div class="charts-grid">
            <div class="chart-card"><h4>🍜 Meal Completion Breakdown</h4><canvas id="chart-serve-per-customer"></canvas></div>
            <div class="chart-card"><h4>⏱️ Eating Duration Distribution</h4><canvas id="chart-serve-histogram"></canvas></div>
        </div>`;
    } else if (agent === 'plating_time' && results.length > 0) {
        chartsHtml = `<div class="charts-grid">
            <div class="chart-card"><h4>📊 Plating Time Distribution (10s buckets)</h4><canvas id="chart-plating-histogram"></canvas></div>
            <div class="chart-card"><h4>📈 Plating Time by Event Start</h4><canvas id="chart-plating-by-time"></canvas></div>
        </div>`;
    } else if (agent === 'noodle_rotation' && results.length > 0) {
        chartsHtml = `<div class="charts-grid">
            <div class="chart-card"><h4>🔄 Rotation Count per Event</h4><canvas id="chart-noodle-per-event"></canvas></div>
            <div class="chart-card"><h4>📊 Rotation Count Distribution</h4><canvas id="chart-noodle-histogram"></canvas></div>
        </div>`;
    } else if (agent === 'bowl_completion_rate' && results.length > 0) {
        chartsHtml = `<div class="charts-grid">
            <div class="chart-card"><h4>🟢 Completion Rate Breakdown</h4><canvas id="chart-bowl-donut"></canvas></div>
            <div class="chart-card"><h4>📈 Detection Confidence by Event</h4><canvas id="chart-bowl-confidence"></canvas></div>
        </div>`;
    } else if (agent === 'pork_weighing') {
        const hasWeightData = results.some(r => (r.verified_reading ?? r.phase1_reading) !== null);
        if (hasWeightData) {
            chartsHtml = `<div class="charts-grid">
                <div class="chart-card"><h4>📊 Weight Distribution</h4><canvas id="chart-histogram"></canvas></div>
                <div class="chart-card"><h4>📈 Weight Over Time</h4><canvas id="chart-timeseries"></canvas></div>
            </div>`;
        }
    }

    // ── Agent-specific detection card rows ──────────────────────────────────
    function detectionCardHtml(r) {
        const p2on = job.phase2_enabled !== false;
        const confBadge = `<span class="detection-confidence" style="color:${getConfidenceColor(r.confidence)}">${(r.confidence * 100).toFixed(0)}% confidence</span>`;
        const tsSpan = r.real_timestamp ? `<span style="font-size:0.82rem;color:var(--accent-cyan);font-family:monospace;">🕐 ${r.real_timestamp}</span>` : '';

        // Match frames for this event.
        // Handles three filename conventions:
        //   1. Subdirectory: 'event_1/frame.png', 'detection_001/frame.png' (pork, noodle, bowl)
        //   2. Flat file:    'phase1_event1_start_12.3s_bowl.png' (plating, serve_time)
        //   3. Flat file (phase2): 'phase2_event1_end_45.0s_customer.png'
        const rFrames = (job.framesData || []).filter(f => {
            if (!f.filename) return false;
            const dirMatch = f.filename.match(/^(?:event|detection)_0*(\d+)\//i);
            if (dirMatch) {
                return parseInt(dirMatch[1], 10) === parseInt(r.event_id, 10);
            }
            const flatMatch = f.filename.match(/^phase\d+_event(\d+)_/i);
            if (flatMatch) {
                return parseInt(flatMatch[1], 10) === parseInt(r.event_id, 10);
            }
            return false;
        });

        const framesHtml = rFrames.length ? `
            <div style="margin-top: 0.75rem;">
                <h4 style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.4rem;">🖼️ Verification Frames</h4>
                <div style="display: flex; gap: 0.5rem; overflow-x: auto; padding-bottom: 0.5rem;" class="frames-mini-gallery">
                    ${rFrames.map(f => {
            const src = f.url ? f.url : `data:${f.mime};base64,${f.data}`;
            return `<img src="${src}" alt="${escapeHtml(f.filename)}" style="height: 100px; border-radius: 4px; max-width: 200px; object-fit: cover; cursor: pointer;" title="Click to view full screen" loading="lazy" onclick="openImageModal(this.src)">`;
        }).join('')}
                </div>
            </div>` : '';

        if (agent === 'serve_time') {
            const h = job.recording_hour;
            const fmt = (s) => (h != null) ? computeRealHms(s, h) : `${(s || 0).toFixed(1)}s`;
            const eatDur = r.verified_reading != null ? `${Math.floor(r.verified_reading / 60)}m ${Math.round(r.verified_reading % 60)}s` : '—';
            return `<div class="detection-card">
                <div class="detection-header">
                    <span class="detection-id">Customer #${r.event_id}</span>
                    <div style="display:flex;gap:0.75rem;align-items:center;">${tsSpan}${confBadge}</div>
                </div>
                <div class="detection-details">
                    <div class="detail-item"><span class="detail-label">Eating Duration</span><span class="detail-value" style="font-weight:700;color:var(--accent-cyan);">${eatDur}</span></div>
                    <div class="detail-item"><span class="detail-label">Video Time</span><span class="detail-value">${fmt(r.start_time)} — ${fmt(r.end_time)}</span></div>
                </div>
                ${r.description ? `<div style="margin-top:0.5rem;font-size:0.82rem;color:var(--text-secondary);">${escapeHtml(r.description)}</div>` : ''}
                ${framesHtml}
            </div>`;
        } else if (agent === 'plating_time') {
            const h = job.recording_hour;
            const fmt = (s) => (h != null) ? computeRealHms(s, h) : `${(s || 0).toFixed(1)}s`;
            const ptSec = r.verified_reading != null ? `${r.verified_reading.toFixed(1)}s` : '—';
            return `<div class="detection-card">
                <div class="detection-header">
                    <span class="detection-id">Bowl Plating #${r.event_id}</span>
                    <div style="display:flex;gap:0.75rem;align-items:center;">${tsSpan}${confBadge}</div>
                </div>
                <div class="detection-details">
                    <div class="detail-item"><span class="detail-label">Bowl</span><span class="detail-value">${escapeHtml(r.scale || '—')}</span></div>
                    <div class="detail-item"><span class="detail-label">Plating Time</span><span class="detail-value" style="font-weight:700;color:var(--accent-cyan);">${ptSec}</span></div>
                    <div class="detail-item"><span class="detail-label">Video Time</span><span class="detail-value">${fmt(r.start_time)} — ${fmt(r.end_time)}</span></div>
                </div>
                ${r.description ? `<div style="margin-top:0.5rem;font-size:0.82rem;color:var(--text-secondary);">${escapeHtml(r.description)}</div>` : ''}
                ${framesHtml}
            </div>`;
        } else if (agent === 'noodle_rotation') {
            //const verifiedCount = r.verified_reading ?? r.scale ?? '—';
            const phase1Count = r.scale ?? '—';
            const durationSec = (r.end_time && r.start_time) ? (r.end_time - r.start_time) : null;
            const duration = durationSec != null ? `${durationSec.toFixed(1)}s` : '—';
            const p2on = job.phase2_enabled !== false;
            const countColor = 'var(--accent-cyan)';
            // Prefer real HH:MM:SS timestamps; fall back to raw seconds
            const startHms = r.real_timestamp || secondsToHms(r.start_time || 0);
            const endHms = r.real_timestamp
                ? addSecondsToHms(r.real_timestamp, durationSec != null ? durationSec : 0)
                : secondsToHms(r.end_time || 0);
            return `<div class="detection-card">
                <div class="detection-header">
                    <span class="detection-id">Event #${r.event_id}</span>
                    ${confBadge}
                </div>
                <div class="detection-details">
                    <div class="detail-item"><span class="detail-label">Start Time</span><span class="detail-value" style="font-family:monospace;">${r.real_timestamp || (job.recording_hour != null ? computeRealHms(r.start_time, job.recording_hour) : secondsToHms(r.start_time || 0))}</span></div>
                    <div class="detail-item"><span class="detail-label">End Time</span><span class="detail-value" style="font-family:monospace;">${r.real_timestamp ? addSecondsToHms(r.real_timestamp, durationSec != null ? durationSec : 0) : (job.recording_hour != null ? computeRealHms(r.end_time, job.recording_hour) : secondsToHms(r.end_time || 0))}</span></div>
                    <div class="detail-item"><span class="detail-label">Duration</span><span class="detail-value">${duration}</span></div>
                    <div class="detail-item"><span class="detail-label">${p2on ? 'Verified Count' : 'Transfer Count'}</span><span class="detail-value" style="font-weight:700;color:${countColor};">${verifiedCount}</span></div>
                    ${p2on ? `<div class="detail-item"><span class="detail-label">Initial Count</span><span class="detail-value">${phase1Count}</span></div>` : ''}
                </div>
                ${r.description ? `<div style="margin-top:0.5rem;font-size:0.82rem;color:var(--text-secondary);">${escapeHtml(r.description)}</div>` : ''}
                ${framesHtml}
            </div>`;
        } else if (agent === 'bowl_completion_rate') {
            const isComp = r.scale === 'COMPLETED';
            const statusColor = isComp ? 'var(--accent-green)' : 'var(--accent-red)';
            return `<div class="detection-card">
                <div class="detection-header">
                    <span class="detection-id">Event #${r.event_id}</span>
                    <div style="display:flex;gap:0.75rem;align-items:center;">${tsSpan}${confBadge}</div>
                </div>
                <div class="detection-details">
                    <div class="detail-item"><span class="detail-label">Status</span><span class="detail-value" style="font-weight:700;color:${statusColor};">${escapeHtml(r.scale || '—')}</span></div>
                    <div class="detail-item"><span class="detail-label">Video Time</span><span class="detail-value">${job.recording_hour != null ? computeRealHms(r.start_time, job.recording_hour) : secondsToHms(r.start_time || 0)}</span></div>
                </div>
                ${r.description ? `<div style="margin-top:0.5rem;font-size:0.82rem;color:var(--text-secondary);">${escapeHtml(r.description)}</div>` : ''}
                ${framesHtml}
            </div>`;
        } else {
            // pork_weighing
            const h = job.recording_hour;
            const fmt = (s) => (h != null) ? computeRealHms(s, h) : `${(s || 0).toFixed(1)}s`;
            const readingNum = parseFloat(p2on ? (r.verified_reading ?? r.phase1_reading ?? 0) : (r.phase1_reading ?? 0));
            const isStandard = (Math.abs(readingNum - 60) <= 5) || (Math.abs(readingNum - 120) <= 5);
            const portionLabel = isStandard ? 'Standard' : 'Non-Standard';
            const portionColor = isStandard ? 'var(--accent-green)' : 'var(--accent-red)';
            const readingColor = p2on ? 'var(--accent-cyan)' : 'var(--accent-amber)';
            const readingVal = p2on ? (r.verified_reading ?? r.phase1_reading ?? '—') : (r.phase1_reading ?? '—');

            return `<div class="detection-card">
                <div class="detection-header">
                    <span class="detection-id">Pork Weighing #${r.event_id}</span>
                    <div style="display:flex;gap:0.75rem;align-items:center;">${tsSpan}${confBadge}</div>
                </div>
                <div class="detection-details">
                    <div class="detail-item"><span class="detail-label">Portion</span><span class="detail-value" style="font-weight:700;color:${portionColor};">${portionLabel}</span></div>
                    <div class="detail-item"><span class="detail-label">Reading</span><span class="detail-value" style="font-weight:700;color:${readingColor};">${readingVal} ${r.unit || 'g'}</span></div>
                    <div class="detail-item"><span class="detail-label">Video Time</span><span class="detail-value">${fmt(r.start_time)} — ${fmt(r.end_time)}</span></div>
                    <div class="detail-item"><span class="detail-label">State</span><span class="detail-value">${r.reading_state || '—'}</span></div>
                </div>
                ${r.description ? `<div style="margin-top:0.5rem;font-size:0.82rem;color:var(--text-secondary);">${escapeHtml(r.description)}</div>` : ''}
                ${r.reading_correction ? `<div style="margin-top:0.35rem;font-size:0.78rem;color:var(--accent-amber);">📝 ${escapeHtml(r.reading_correction)}</div>` : ''}
                ${framesHtml}
            </div>`;
        }
    }

    const p2on = job.phase2_enabled !== false;
    const evtTitle = p2on ? '✅ Verified Events' : '🔎 Detected Events';
    const emptyMsg = p2on ? 'No events were verified.' : 'No events detected.';

    // ── CLAHE preview gallery (pork_weighing only) ───────────────────────────
    const claheFrames = (job.framesData || []).filter(f => f.filename && f.filename.startsWith('clahe_preview/'));
    const claheGalleryHtml = (agent === 'pork_weighing' && claheFrames.length > 0) ? `
        <div style="margin:1.5rem 0 1rem;">
            <h3 style="font-size:1rem; margin-bottom:0.75rem;">🎞️ Video Frames (Enhanced)</h3>
            <div style="display:flex; gap:0.5rem; overflow-x:auto; padding-bottom:0.5rem; flex-wrap:nowrap;">
                ${claheFrames.map(f => {
        const src = f.url ? f.url : `data:${f.mime};base64,${f.data}`;
        const label = f.filename.replace('clahe_preview/', '').replace(/\.png$/i, '');
        return `<div style="flex:0 0 auto; text-align:center;">
                        <img src="${src}" alt="${escapeHtml(label)}" loading="lazy"
                             style="height:130px; border-radius:4px; border:1px solid var(--border-color); cursor:pointer; display:block;"
                             title="${escapeHtml(label)}" onclick="openImageModal(this.src)">
                        <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.25rem;">${escapeHtml(label)}</div>
                    </div>`;
    }).join('')}
            </div>
        </div>` : '';

    panel.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h2>📋 Results — Job #${job.id}: ${escapeHtml(job.video_name)}</h2>
                <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
                    ${job.recording_date ? `<span class="step-badge">📅 ${job.recording_date} ${job.recording_hour !== null ? String(job.recording_hour).padStart(2, '0') + ':xx' : ''}</span>` : ''}
                    <span class="status-badge status-${job.status}">${getStatusIcon(job.status)} ${job.status}</span>
                    <button class="btn btn-secondary btn-sm" onclick="closeResults()">✕ Close</button>
                </div>
            </div>
            <div class="card-body">
                ${job.error_message ? `
                    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:1rem;margin-bottom:1rem;color:var(--accent-red);">
                        <strong>Error:</strong> ${escapeHtml(job.error_message)}
                    </div>` : ''}

                <div class="results-stats">
                    <div class="stat-card"><div class="stat-value">${(tokenSummary.total_tokens || job.total_tokens || 0).toLocaleString()}</div><div class="stat-label">Total Tokens</div></div>
                    <div class="stat-card"><div class="stat-value">${formatTime(job.created_at)}</div><div class="stat-label">Created</div></div>
                    ${agentKpiHtml}
                </div>

                ${chartsHtml}

                ${claheGalleryHtml}

                ${results.length > 0 ? `
                    <h3 style="margin:1.5rem 0 1rem; font-size:1rem;">${evtTitle}</h3>
                    ${results.map(r => detectionCardHtml(r)).join('')}
                ` : `
                    <div class="empty-state" style="padding:2rem;">
                        <div class="icon">${job.status === 'completed' ? '🔍' : '⏳'}</div>
                        <h3>${job.status === 'completed' ? 'No events detected' : 'Processing...'}</h3>
                        <p>${job.status === 'completed' ? emptyMsg : job.progress_message || 'Results will appear here once complete.'}</p>
                    </div>`}

            </div>
        </div>`;

    // Render charts after DOM update
    setTimeout(() => {
        if (agent === 'serve_time') {
            renderServeTimeCharts(results);
        } else if (agent === 'plating_time') {
            renderPlatingTimeCharts(results);
        } else if (agent === 'noodle_rotation') {
            renderNoodleRotationCharts(results);
        } else if (agent === 'bowl_completion_rate') {
            renderBowlCompletionCharts(results);
        } else {
            const chartData = results
                .filter(r => (r.verified_reading ?? r.phase1_reading) !== null)
                .map(r => ({
                    time: r.real_timestamp || `${(r.start_time || 0).toFixed(0)}s`,
                    weight: parseFloat(r.verified_reading ?? r.phase1_reading),
                }))
                .filter(d => !isNaN(d.weight));
            if (chartData.length > 0) {
                renderHistogram(chartData);
                renderTimeSeries(chartData);
            }
        }
    }, 50);
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// =============================================================================
// Charts
// =============================================================================
function destroyChart(id) {
    if (state.activeCharts[id]) { state.activeCharts[id].destroy(); delete state.activeCharts[id]; }
}

function renderHistogram(chartData) {
    destroyChart('chart-histogram');
    const canvas = $('#chart-histogram');
    if (!canvas) return;
    const weights = chartData.map(d => d.weight);
    const min = Math.floor(Math.min(...weights));
    const max = Math.ceil(Math.max(...weights));
    const binSize = Math.max(1, Math.round((max - min) / 10));
    const bins = {};
    for (let b = min; b <= max; b += binSize) bins[b] = 0;
    weights.forEach(w => { const b = Math.floor(w / binSize) * binSize; bins[b] = (bins[b] || 0) + 1; });
    state.activeCharts['chart-histogram'] = new Chart(canvas, {
        type: 'bar',
        data: { labels: Object.keys(bins).map(b => `${b}–${+b + binSize}`), datasets: [{ label: 'Count', data: Object.values(bins), backgroundColor: 'rgba(59,130,246,0.6)', borderColor: 'rgba(59,130,246,1)', borderWidth: 1, borderRadius: 4 }] },
        options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Weight (g)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }, y: { title: { display: true, text: 'Count', color: '#94a3b8' }, ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true } } }
    });
}

function renderTimeSeries(chartData) {
    destroyChart('chart-timeseries');
    const canvas = $('#chart-timeseries');
    if (!canvas) return;
    state.activeCharts['chart-timeseries'] = new Chart(canvas, {
        type: 'bar',
        data: { labels: chartData.map(d => d.time), datasets: [{ label: 'Weight (g)', data: chartData.map(d => d.weight), backgroundColor: 'rgba(139,92,246,0.6)', borderColor: 'rgba(139,92,246,1)', borderWidth: 1, borderRadius: 4 }] },
        options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Time', color: '#94a3b8' }, ticks: { color: '#94a3b8', maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.05)' } }, y: { title: { display: true, text: 'Weight (g)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: false } } }
    });
}

function renderServeTimeCharts(results) {
    // Chart 1: service time per customer event (bar)
    destroyChart('chart-serve-per-customer');
    const perCustCanvas = $('#chart-serve-per-customer');
    if (perCustCanvas) {
        const data = results
            .filter(r => r.verified_reading != null)
            .map(r => ({
                label: r.real_timestamp || `Evt ${r.event_id}`,
                value: r.verified_reading,
            }));
        state.activeCharts['chart-serve-per-customer'] = new Chart(perCustCanvas, {
            type: 'bar',
            data: {
                labels: data.map(d => d.label),
                datasets: [{
                    label: 'Service Time (s)',
                    data: data.map(d => d.value),
                    backgroundColor: 'rgba(34,197,94,0.6)',
                    borderColor: 'rgba(34,197,94,1)',
                    borderWidth: 1,
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Customer', color: '#94a3b8' }, ticks: { color: '#94a3b8', maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Service Time (s)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                },
            },
        });
    }

    // Chart 2: service time distribution histogram (20s buckets)
    destroyChart('chart-serve-histogram');
    const histCanvas = $('#chart-serve-histogram');
    if (histCanvas) {
        const values = results.map(r => r.verified_reading).filter(v => v != null);
        const bucketSize = 20;
        const min = Math.floor(Math.min(...values) / bucketSize) * bucketSize;
        const max = Math.ceil(Math.max(...values) / bucketSize) * bucketSize;
        const bins = {};
        for (let b = min; b <= max; b += bucketSize) bins[b] = 0;
        values.forEach(v => { const b = Math.floor(v / bucketSize) * bucketSize; bins[b] = (bins[b] || 0) + 1; });
        state.activeCharts['chart-serve-histogram'] = new Chart(histCanvas, {
            type: 'bar',
            data: {
                labels: Object.keys(bins).map(b => `${b}–${+b + bucketSize}s`),
                datasets: [{ label: 'Count', data: Object.values(bins), backgroundColor: 'rgba(34,197,94,0.6)', borderColor: 'rgba(34,197,94,1)', borderWidth: 1, borderRadius: 4 }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Service Time (s)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Count', color: '#94a3b8' }, ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                },
            },
        });
    }
}

function renderPlatingTimeCharts(results) {
    // Chart 1: plating time distribution histogram (10s buckets)
    destroyChart('chart-plating-histogram');
    const histCanvas = $('#chart-plating-histogram');
    if (histCanvas) {
        const values = results.map(r => r.verified_reading).filter(v => v != null);
        const bucketSize = 10;
        const min = Math.floor(Math.min(...values) / bucketSize) * bucketSize;
        const max = Math.ceil(Math.max(...values) / bucketSize) * bucketSize;
        const bins = {};
        for (let b = min; b <= max; b += bucketSize) bins[b] = 0;
        values.forEach(v => { const b = Math.floor(v / bucketSize) * bucketSize; bins[b] = (bins[b] || 0) + 1; });
        state.activeCharts['chart-plating-histogram'] = new Chart(histCanvas, {
            type: 'bar',
            data: {
                labels: Object.keys(bins).map(b => `${b}–${+b + bucketSize}s`),
                datasets: [{ label: 'Count', data: Object.values(bins), backgroundColor: 'rgba(251,191,36,0.6)', borderColor: 'rgba(251,191,36,1)', borderWidth: 1, borderRadius: 4 }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Plating Time (s)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Count', color: '#94a3b8' }, ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                },
            },
        });
    }

    // Chart 2: plating time by event start timestamp (sorted)
    destroyChart('chart-plating-by-time');
    const byTimeCanvas = $('#chart-plating-by-time');
    if (byTimeCanvas) {
        const sorted = [...results]
            .filter(r => r.verified_reading != null)
            .sort((a, b) => (a.start_time || 0) - (b.start_time || 0));
        state.activeCharts['chart-plating-by-time'] = new Chart(byTimeCanvas, {
            type: 'bar',
            data: {
                labels: sorted.map(r => r.real_timestamp || `${(r.start_time || 0).toFixed(0)}s`),
                datasets: [{
                    label: 'Plating Time (s)',
                    data: sorted.map(r => r.verified_reading),
                    backgroundColor: 'rgba(251,191,36,0.6)',
                    borderColor: 'rgba(251,191,36,1)',
                    borderWidth: 1,
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Event Start Time', color: '#94a3b8' }, ticks: { color: '#94a3b8', maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Plating Time (s)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                },
            },
        });
    }
}

function renderNoodleRotationCharts(results) {
    const sorted = [...results]
        .filter(r => (r.verified_reading ?? r.scale) != null)
        .sort((a, b) => (a.start_time || 0) - (b.start_time || 0));

    // Chart 1: rotation count per event (bar, sorted by video time)
    destroyChart('chart-noodle-per-event');
    const perEventCanvas = $('#chart-noodle-per-event');
    if (perEventCanvas && sorted.length > 0) {
        const labels = sorted.map(r => r.real_timestamp || `${(r.start_time || 0).toFixed(0)}s`);
        const counts = sorted.map(r => Number(r.verified_reading ?? r.scale));
        state.activeCharts['chart-noodle-per-event'] = new Chart(perEventCanvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Rotations',
                    data: counts,
                    backgroundColor: 'rgba(139,92,246,0.65)',
                    borderColor: 'rgba(139,92,246,1)',
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Event Time', color: '#94a3b8' }, ticks: { color: '#94a3b8', maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Rotation Count', color: '#94a3b8' }, ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true }
                }
            }
        });
    }

    // Chart 2: rotation count distribution histogram (1-rotation buckets)
    destroyChart('chart-noodle-histogram');
    const histCanvas = $('#chart-noodle-histogram');
    if (histCanvas && sorted.length > 0) {
        const values = sorted.map(r => Number(r.verified_reading ?? r.scale));
        const min = Math.min(...values);
        const max = Math.max(...values);
        const bins = {};
        for (let b = min; b <= max; b++) bins[b] = 0;
        values.forEach(v => { bins[v] = (bins[v] || 0) + 1; });
        state.activeCharts['chart-noodle-histogram'] = new Chart(histCanvas, {
            type: 'bar',
            data: {
                labels: Object.keys(bins).map(b => `${b} rot`),
                datasets: [{
                    label: 'Events',
                    data: Object.values(bins),
                    backgroundColor: 'rgba(59,130,246,0.65)',
                    borderColor: 'rgba(59,130,246,1)',
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Rotation Count', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Number of Events', color: '#94a3b8' }, ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true }
                }
            }
        });
    }
}

function renderBowlCompletionCharts(results) {
    // Chart 1: completion rate donut
    destroyChart('chart-bowl-donut');
    const donutCanvas = $('#chart-bowl-donut');
    if (donutCanvas) {
        const completed = results.filter(r => r.scale === 'COMPLETED').length;
        const notCompleted = results.length - completed;
        state.activeCharts['chart-bowl-donut'] = new Chart(donutCanvas, {
            type: 'doughnut',
            data: {
                labels: ['Completed', 'Not Completed'],
                datasets: [{
                    data: [completed, notCompleted],
                    backgroundColor: ['rgba(34,197,94,0.8)', 'rgba(239,68,68,0.8)'],
                    borderColor: ['rgba(34,197,94,1)', 'rgba(239,68,68,1)'],
                    borderWidth: 2,
                }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#94a3b8' } },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const pct = results.length > 0 ? ((ctx.parsed / results.length) * 100).toFixed(1) : 0;
                                return ` ${ctx.label}: ${ctx.parsed} (${pct}%)`;
                            },
                        },
                    },
                },
            },
        });
    }

    // Chart 2: detection confidence by event (sorted by video time)
    destroyChart('chart-bowl-confidence');
    const confCanvas = $('#chart-bowl-confidence');
    if (confCanvas) {
        const sorted = [...results].sort((a, b) => (a.start_time || 0) - (b.start_time || 0));
        const bgColors = sorted.map(r => r.scale === 'COMPLETED' ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)');
        const borderColors = sorted.map(r => r.scale === 'COMPLETED' ? 'rgba(34,197,94,1)' : 'rgba(239,68,68,1)');
        state.activeCharts['chart-bowl-confidence'] = new Chart(confCanvas, {
            type: 'bar',
            data: {
                labels: sorted.map(r => r.real_timestamp || `${(r.start_time || 0).toFixed(0)}s`),
                datasets: [{
                    label: 'Confidence',
                    data: sorted.map(r => (r.confidence * 100).toFixed(1)),
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => ` ${sorted[ctx.dataIndex]?.scale || ''}: ${ctx.parsed.y}% confidence` } },
                },
                scales: {
                    x: { title: { display: true, text: 'Detection Time', color: '#94a3b8' }, ticks: { color: '#94a3b8', maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Confidence (%)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true, max: 100 },
                },
            },
        });
    }
}

// =============================================================================
// Verification Frames
// =============================================================================
async function loadVerificationFrames(jobId) {
    const container = $('#frames-gallery-container');
    if (!container) return;
    container.innerHTML = '<span class="spinner"></span> Loading frames...';
    try {
        const resp = await fetch(`/api/jobs/${jobId}/frames`);
        if (!resp.ok) throw new Error('Failed to load frames');
        const data = await resp.json();
        if (!data.frames?.length) {
            container.innerHTML = '<span style="color:var(--text-muted);font-size:0.85rem;">No verification frames found.</span>';
            return;
        }
        container.innerHTML = `<div class="frames-gallery">${data.frames.map(f => {
            const src = f.url ? f.url : `data:${f.mime};base64,${f.data}`;
            return `<div class="frame-thumb">
                <img src="${src}" alt="${escapeHtml(f.filename)}" loading="lazy">
                <div class="frame-label">${escapeHtml(f.filename)}</div>
            </div>`;
        }).join('')}</div>`;
    } catch (err) {
        container.innerHTML = `<span style="color:var(--accent-red);font-size:0.85rem;">Error: ${err.message}</span>`;
    }
}

function getConfidenceColor(c) {
    return c >= 0.8 ? 'var(--accent-green)' : c >= 0.5 ? 'var(--accent-amber)' : 'var(--accent-red)';
}

function closeResults() {
    const panel = $('#results-panel');
    if (panel) { panel.classList.remove('active'); panel.innerHTML = ''; }
    Object.keys(state.activeCharts).forEach(id => state.activeCharts[id].destroy());
    state.activeCharts = {};
    state.activeJobId = null;
}

// =============================================================================
// Delete Job
// =============================================================================
async function deleteJob(jobId) {
    if (!confirm(`Delete Job #${jobId}? This cannot be undone.`)) return;

    const jobIdStr = String(jobId);

    // Mark as pending so loadJobs() (polling) won't resurrect it during the request
    state.pendingDeletes.add(jobIdStr);

    // Optimistically remove from UI immediately
    state.jobs = state.jobs.filter(j => String(j.id) !== jobIdStr);
    renderJobsTable();
    if (String(state.activeJobId) === jobIdStr) closeResults();

    try {
        const resp = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to delete job');
        }
        showToast(`Job #${jobId} deleted.`, 'success');
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    } finally {
        // Always clear the pending flag and sync with server state
        state.pendingDeletes.delete(jobIdStr);
        await loadJobs();
    }
}
