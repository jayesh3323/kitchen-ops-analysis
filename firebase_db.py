"""
Firebase Firestore database backend.

Mirrors the interface of database.py (Job, Result, TokenUsage, init_db,
SessionLocal) so that worker.py and main.py need zero changes when
USE_FIREBASE=true is set in the environment.

Collections:
  jobs/             — one document per job
  jobs/{id}/results — detection results sub-collection
  jobs/{id}/token_usages — token usage sub-collection

Authentication:
  - FIREBASE_SERVICE_ACCOUNT_PATH  →  local JSON key file
  - FIREBASE_SERVICE_ACCOUNT_JSON  →  raw JSON string (for HF Spaces Secrets)
  - Neither set                    →  Application Default Credentials

SQLAlchemy-style filter emulation:
  FirestoreJob, FirestoreResult, etc. expose class-level _FirestoreColumn
  descriptors so that expressions like  Job.id == 5  and  Job.status == "queued"
  produce _FirestoreFilter objects rather than Python booleans.
  FirestoreSession.query().filter(expr).first()/all() interprets these filters
  and translates them to Firestore queries.
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import config as app_config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy-style column descriptors
# ─────────────────────────────────────────────────────────────────────────────

class _FirestoreColumn:
    """
    Class-level descriptor that enables expressions like:
       Job.id == 5          →  _FirestoreFilter("id", "==", 5)
       Job.status == "q"    →  _FirestoreFilter("status", "==", "q")
    """
    def __init__(self, field_name: str):
        self.field_name = field_name

    # desc() / asc() for order_by compatibility
    def desc(self):
        return _FirestoreOrder(self.field_name, descending=True)

    def asc(self):
        return _FirestoreOrder(self.field_name, descending=False)

    def __eq__(self, value):
        return _FirestoreFilter(self.field_name, "==", value)

    def __ne__(self, value):
        return _FirestoreFilter(self.field_name, "!=", value)

    def __lt__(self, value):
        return _FirestoreFilter(self.field_name, "<", value)

    def __le__(self, value):
        return _FirestoreFilter(self.field_name, "<=", value)

    def __gt__(self, value):
        return _FirestoreFilter(self.field_name, ">", value)

    def __ge__(self, value):
        return _FirestoreFilter(self.field_name, ">=", value)

    # Instance-level: let __get__ return the actual stored value
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self          # class-level access → return descriptor itself
        if self.field_name == "id" and hasattr(obj, "_doc_id"):
            return obj._doc_id
        return obj._data.get(self.field_name)   # instance access → return value

    def __set__(self, obj, value):
        obj._data[self.field_name] = value


class _FirestoreFilter:
    def __init__(self, field: str, op: str, value: Any):
        self.field = field
        self.op = op
        self.value = value

    def to_firestore_filter(self):
        from google.cloud.firestore_v1.base_query import FieldFilter
        return FieldFilter(self.field, self.op, self.value)


class _FirestoreOrder:
    def __init__(self, field: str, descending: bool = True):
        self.field = field
        self.descending = descending


# ─────────────────────────────────────────────────────────────────────────────
# Firestore singleton
# ─────────────────────────────────────────────────────────────────────────────

_firestore_db = None
_firebase_app = None


def _get_idst():
    """Return current UTC time; JS will convert it to local system time."""
    from datetime import timezone
    return datetime.now(timezone.utc)


def _init_firebase():
    """Initialize Firebase Admin SDK and return the Firestore client."""
    global _firestore_db, _firebase_app

    if _firestore_db is not None:
        return _firestore_db

    import firebase_admin
    from firebase_admin import credentials, firestore

    if firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:
        _firebase_app = firebase_admin.get_app()
    else:
        _sa_json = (app_config.FIREBASE_SERVICE_ACCOUNT_JSON or "").strip()
        if _sa_json:
            try:
                sa_dict = json.loads(_sa_json)
                # Fix literal escaped newlines that happen when pasting JSON into HF Secrets
                if "private_key" in sa_dict:
                    sa_dict["private_key"] = sa_dict["private_key"].replace("\\n", "\n")
                cred = credentials.Certificate(sa_dict)
                logger.info("Firebase: using service account from FIREBASE_SERVICE_ACCOUNT_JSON")
            except json.JSONDecodeError as e:
                raise RuntimeError(f"FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")
        elif app_config.FIREBASE_SERVICE_ACCOUNT_PATH:
            cred = credentials.Certificate(app_config.FIREBASE_SERVICE_ACCOUNT_PATH)
            logger.info(f"Firebase: using service account file: {app_config.FIREBASE_SERVICE_ACCOUNT_PATH}")
        else:
            cred = credentials.ApplicationDefault()
            logger.info("Firebase: using Application Default Credentials")

        options: Dict[str, str] = {}
        if app_config.FIREBASE_PROJECT_ID:
            options["projectId"] = app_config.FIREBASE_PROJECT_ID
        if app_config.FIREBASE_STORAGE_BUCKET:
            options["storageBucket"] = app_config.FIREBASE_STORAGE_BUCKET

        try:
            _firebase_app = firebase_admin.initialize_app(cred, options or None)
        except ValueError:
            # App was already initialized (e.g. by main.py startup); just reuse it.
            _firebase_app = firebase_admin.get_app()

    _firestore_db = firestore.client()
    logger.info("Firestore client initialized.")
    return _firestore_db


def _get_db():
    return _init_firebase()


# ─────────────────────────────────────────────────────────────────────────────
# Model proxy classes (mirror SQLAlchemy models)
# ─────────────────────────────────────────────────────────────────────────────

class FirestoreJob:
    """
    Proxy that behaves like a SQLAlchemy Job row.
    Class-level _FirestoreColumn descriptors allow Job.id == 5 style filters.
    Instance attribute access / mutation routes through self._data dict.
    """
    # Class-level column descriptors — enable SQLAlchemy-style filter expressions
    id              = _FirestoreColumn("id")          # the Firestore doc ID
    video_name      = _FirestoreColumn("video_name")
    video_path      = _FirestoreColumn("video_path")
    status          = _FirestoreColumn("status")
    created_at      = _FirestoreColumn("created_at")
    updated_at      = _FirestoreColumn("updated_at")
    error_message   = _FirestoreColumn("error_message")
    roi_coords      = _FirestoreColumn("roi_coords")
    timestamp_region_coords = _FirestoreColumn("timestamp_region_coords")
    config_json     = _FirestoreColumn("config_json")
    agent           = _FirestoreColumn("agent")
    progress_message = _FirestoreColumn("progress_message")
    phase1_count    = _FirestoreColumn("phase1_count")
    phase2_count    = _FirestoreColumn("phase2_count")
    total_tokens    = _FirestoreColumn("total_tokens")
    recording_date  = _FirestoreColumn("recording_date")
    recording_hour  = _FirestoreColumn("recording_hour")
    output_dir      = _FirestoreColumn("output_dir")

    def __init__(self, doc_id: str = None, data: Dict[str, Any] = None, db=None, **kwargs):
        # Bypass __setattr__ to set private attrs
        object.__setattr__(self, "_doc_id", doc_id)
        object.__setattr__(self, "_data", data if data is not None else {})
        object.__setattr__(self, "_db", db or _get_db())
        # Accept SQLAlchemy-style keyword constructor args (e.g. Job(video_name=...))
        for k, v in kwargs.items():
            self._data[k] = v
        
        # Firestore strictly requires fields used in order_by to exist.
        if "created_at" not in self._data:
            self._data["created_at"] = _get_idst()

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __getattr__(self, name):
        # Fallback for attrs not covered by descriptors
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    @property
    def _id(self):
        """Internal: the Firestore document ID."""
        return self._doc_id

    def to_dict(self):
        d = dict(self._data)
        d["id"] = self._doc_id          # Expose as integer for API compatibility
        # Normalise datetime → ISO string
        for key in ("created_at", "updated_at"):
            v = d.get(key)
            if isinstance(v, datetime):
                d[key] = v.isoformat()
        # Decode JSON-encoded coordinate strings
        for key in ("roi_coords", "timestamp_region_coords"):
            v = d.get(key)
            if isinstance(v, str):
                try:
                    d[key] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def _save(self):
        """Persist the current in-memory _data dict to Firestore."""
        self._data["updated_at"] = _get_idst()
        if self._doc_id is None:
            # Auto-generate ID on first save
            _, doc_ref = self._db.collection("jobs").add(self._data)
            object.__setattr__(self, "_doc_id", doc_ref.id)
        else:
            self._db.collection("jobs").document(str(self._doc_id)).set(self._data)

    # Lazy relationship accessors
    @property
    def results(self):
        return _list_results(self._doc_id)

    @property
    def token_usages(self):
        return _list_token_usages(self._doc_id)


class FirestoreResult:
    """Proxy that behaves like a SQLAlchemy Result row."""

    def __init__(self, doc_id: str = None, data: Dict[str, Any] = None, **kwargs):
        object.__setattr__(self, "_doc_id", doc_id)
        object.__setattr__(self, "_data", data if data is not None else {})
        for k, v in kwargs.items():
            self._data[k] = v

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    @property
    def id(self):
        return self._doc_id

    def to_dict(self):
        d = dict(self._data)
        d["id"] = self._doc_id
        return d


class FirestoreTokenUsage:
    """Proxy that behaves like a SQLAlchemy TokenUsage row."""

    def __init__(self, doc_id: str = None, data: Dict[str, Any] = None, **kwargs):
        object.__setattr__(self, "_doc_id", doc_id)
        object.__setattr__(self, "_data", data if data is not None else {})
        for k, v in kwargs.items():
            self._data[k] = v

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    @property
    def id(self):
        return self._doc_id

    @property
    def total_tokens(self):
        return self._data.get("total_tokens", 0)

    @property
    def phase(self):
        return self._data.get("phase", "")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-collection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _list_results(job_id: str) -> List[FirestoreResult]:
    db = _get_db()
    docs = db.collection("jobs").document(str(job_id)).collection("results").stream()
    return [FirestoreResult(d.id, d.to_dict()) for d in docs]


def _list_token_usages(job_id: str) -> List[FirestoreTokenUsage]:
    db = _get_db()
    docs = db.collection("jobs").document(str(job_id)).collection("token_usages").stream()
    return [FirestoreTokenUsage(d.id, d.to_dict()) for d in docs]


# ─────────────────────────────────────────────────────────────────────────────
# Session — mirrors SQLAlchemy Session interface
# ─────────────────────────────────────────────────────────────────────────────

class FirestoreSession:
    """
    Thin session that mirrors the SQLAlchemy Session API used in
    worker.py and main.py. Write operations are non-transactional
    (Firestore commits are atomic per document).
    """

    def __init__(self):
        self._db = _get_db()
        self._dirty_jobs: List[FirestoreJob] = []     # jobs mutated via setattr
        self._pending_results: List[FirestoreResult] = []
        self._pending_token_usages: List[FirestoreTokenUsage] = []

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, model_class):
        return _FirestoreQuery(model_class, self._db, session=self)

    # ── Write ────────────────────────────────────────────────────────────────

    def add(self, obj):
        """Stage a new Result or TokenUsage for writing on commit()."""
        if isinstance(obj, FirestoreResult):
            self._pending_results.append(obj)
        elif isinstance(obj, FirestoreTokenUsage):
            self._pending_token_usages.append(obj)
        elif isinstance(obj, FirestoreJob):
            # Jobs are committed via _dirty_jobs tracking or _register_dirty
            self._register_dirty(obj)
        else:
            logger.warning(f"FirestoreSession.add: unknown type {type(obj)}")

    def _register_dirty(self, job: FirestoreJob):
        """Mark a job as needing to be saved on commit()."""
        if job not in self._dirty_jobs:
            self._dirty_jobs.append(job)

    def commit(self):
        """Flush all pending writes and dirty jobs to Firestore."""
        import job_cache
        # Save mutated jobs and update cache
        for job in self._dirty_jobs:
            job._save()
            try:
                job_cache.upsert(job.to_dict())
            except Exception:
                pass
        # Save new Result documents; invalidate results cache for those jobs
        result_job_ids = set()
        for result in self._pending_results:
            jid = result._data.get("job_id")
            job_ref = self._db.collection("jobs").document(str(jid))
            job_ref.collection("results").add(result._data)
            if jid is not None:
                result_job_ids.add(jid)
        for jid in result_job_ids:
            try:
                job_cache.invalidate_results(jid)
            except Exception:
                pass
        # Save new TokenUsage documents
        for tu in self._pending_token_usages:
            job_ref = self._db.collection("jobs").document(str(tu._data.get("job_id")))
            job_ref.collection("token_usages").add(tu._data)

        self._dirty_jobs.clear()
        self._pending_results.clear()
        self._pending_token_usages.clear()

    def delete(self, obj):
        """Delete a job and all its sub-collections from Firestore."""
        if isinstance(obj, FirestoreJob):
            job_ref = self._db.collection("jobs").document(str(obj._doc_id))
            for sub in ("results", "token_usages"):
                for d in job_ref.collection(sub).stream():
                    d.reference.delete()
            job_ref.delete()
            try:
                import job_cache
                job_cache.remove(obj._doc_id)
            except Exception:
                pass

    def direct_update_job(self, job_id, **fields):
        """
        Write-only field update using Firestore's update() — no read required.
        Used by worker._update_job to avoid a read-before-write on every
        status/progress change.
        """
        fields["updated_at"] = _get_idst()
        self._db.collection("jobs").document(str(job_id)).update(fields)
        # Keep in-memory cache consistent
        try:
            import job_cache
            cached = job_cache.get(job_id)
            if cached:
                for k, v in fields.items():
                    cached[k] = v.isoformat() if isinstance(v, datetime) else v
                job_cache.upsert(cached)
        except Exception:
            pass

    def rollback(self):
        self._dirty_jobs.clear()
        self._pending_results.clear()
        self._pending_token_usages.clear()

    def refresh(self, obj):
        """No-op: _data is already current after _save(); avoids a redundant read."""
        pass

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Fluent query builder
# ─────────────────────────────────────────────────────────────────────────────

class _FirestoreQuery:
    def __init__(self, model_class, db, session: FirestoreSession = None):
        self._model_class = model_class
        self._db = db
        self._session = session
        self._filters: List[_FirestoreFilter] = []
        self._order: Optional[_FirestoreOrder] = None
        self._limit_n: Optional[int] = None

    def filter(self, condition):
        """Accept _FirestoreFilter expressions produced by column descriptors."""
        if isinstance(condition, _FirestoreFilter):
            self._filters.append(condition)
        elif isinstance(condition, bool):
            # Fallback: SQLAlchemy expression evaluated to bool (shouldn't happen
            # when FirestoreJob descriptors are used correctly, but guard anyway)
            logger.warning("filter() received a plain bool — skipping filter")
        else:
            logger.warning(f"filter() received unsupported expression: {condition!r}")
        return self

    def order_by(self, expr):
        if isinstance(expr, _FirestoreOrder):
            self._order = expr
        elif hasattr(expr, "field_name"):
            # Raw _FirestoreColumn passed without desc()/asc()
            self._order = _FirestoreOrder(expr.field_name, descending=False)
        return self

    def limit(self, n: int):
        self._limit_n = n
        return self

    def first(self):
        results = self._execute(limit=1)
        return results[0] if results else None

    def all(self):
        return self._execute()

    def _execute(self, limit: int = None) -> List[FirestoreJob]:
        col = self._db.collection("jobs")
        
        # Fast path for fetching a single document by its ID
        for f in self._filters:
            if f.field == "id" and f.op == "==":
                doc = col.document(str(f.value)).get()
                if not doc.exists:
                    return []
                job = FirestoreJob(doc_id=doc.id, data=doc.to_dict(), db=self._db)
                # Do NOT register as dirty — queried jobs are read-only.
                # Only jobs staged via session.add() (create path) are saved on commit().
                # Registering here caused commit() to overwrite the job document with
                # its original (stale) data, resetting status back to "queued".
                return [job]

        q = col
        for f in self._filters:
            q = q.where(filter=f.to_firestore_filter())

        if self._order:
            from google.cloud.firestore import Query
            direction = Query.DESCENDING if self._order.descending else Query.ASCENDING
            q = q.order_by(self._order.field, direction=direction)

        effective_limit = limit or self._limit_n
        if effective_limit:
            q = q.limit(effective_limit)

        jobs = []
        for doc in q.stream():
            job = FirestoreJob(doc_id=doc.id, data=doc.to_dict(), db=self._db)
            # Do NOT register as dirty — queried jobs are read-only.
            jobs.append(job)
        return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Session factory (mirrors SQLAlchemy's SessionLocal callable)
# ─────────────────────────────────────────────────────────────────────────────

def FirestoreSessionLocal() -> FirestoreSession:
    return FirestoreSession()


# ─────────────────────────────────────────────────────────────────────────────
# init_db (mirrors database.py's init_db)
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Connect to Firestore and load all jobs into the in-memory cache."""
    db = _get_db()
    try:
        import job_cache
        docs = list(db.collection("jobs").stream())
        job_dicts = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            # Normalise datetime fields to ISO strings (mirrors to_dict())
            for key in ("created_at", "updated_at"):
                v = data.get(key)
                if isinstance(v, datetime):
                    data[key] = v.isoformat()
            # Decode JSON-encoded coord strings
            for key in ("roi_coords", "timestamp_region_coords"):
                v = data.get(key)
                if isinstance(v, str):
                    try:
                        data[key] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        pass
            job_dicts.append(data)
        job_cache.populate(job_dicts)
        logger.info(f"Firestore backend ready. {len(job_dicts)} jobs loaded into cache.")
    except Exception as e:
        logger.warning(f"Firestore init/cache-load failed (non-fatal): {e}")
        # Mark cache as ready even if empty so the app doesn't stall
        try:
            import job_cache as _jc
            _jc.populate([])
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Convenience CRUD helpers (used by advanced callers that know they're on Firebase)
# ─────────────────────────────────────────────────────────────────────────────

def create_job(
    video_name: str,
    video_path: str,
    status: str = "queued",
    roi_coords: Optional[str] = None,
    timestamp_region_coords: Optional[str] = None,
    config_json: Optional[str] = None,
    progress_message: Optional[str] = None,
    recording_hour: Optional[int] = None,
    recording_date: Optional[str] = None,
    agent: Optional[str] = None,
) -> FirestoreJob:
    """Create a new job document and return a FirestoreJob proxy."""
    db = _get_db()
    now = _get_idst()
    data = {
        "video_name": video_name,
        "video_path": video_path,
        "status": status,
        "created_at": now,
        "updated_at": now,
        "roi_coords": roi_coords,
        "timestamp_region_coords": timestamp_region_coords,
        "config_json": config_json,
        "progress_message": progress_message or "Queued for processing...",
        "recording_hour": recording_hour,
        "recording_date": recording_date,
        "agent": agent,
        "phase1_count": 0,
        "phase2_count": 0,
        "total_tokens": 0,
        "error_message": None,
        "output_dir": None,
    }
    _, doc_ref = db.collection("jobs").add(data)
    job = FirestoreJob(doc_id=doc_ref.id, data=data, db=db)
    return job


def update_job(job_id: str, **kwargs) -> Optional[FirestoreJob]:
    """Update fields on an existing job document."""
    db = _get_db()
    ref = db.collection("jobs").document(str(job_id))
    doc = ref.get()
    if not doc.exists:
        logger.error(f"update_job: job '{job_id}' not found in Firestore")
        return None
    data = doc.to_dict()
    data.update(kwargs)
    data["updated_at"] = _get_idst()
    ref.set(data)
    return FirestoreJob(doc_id=job_id, data=data, db=db)


def add_result(job_id: str, result_data: Dict[str, Any]):
    """Append a result to a job's results sub-collection."""
    db = _get_db()
    db.collection("jobs").document(str(job_id)).collection("results").add(result_data)


def add_token_usage(job_id: str, usage_data: Dict[str, Any]):
    """Append a token-usage record to a job's token_usages sub-collection."""
    db = _get_db()
    db.collection("jobs").document(str(job_id)).collection("token_usages").add(usage_data)
