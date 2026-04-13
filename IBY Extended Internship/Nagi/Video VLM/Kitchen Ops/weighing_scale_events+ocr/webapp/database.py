"""
SQLAlchemy database models for the Pork Weighing Analysis Web App.
Tracks jobs, results, and token usage with persistent SQLite storage.
"""
import json
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, Boolean,
    DateTime, ForeignKey, Enum as SAEnum
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config import DATABASE_URL

def get_local_time():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).replace(tzinfo=None)

Base = declarative_base()


class Job(Base):
    """Represents an analysis job submitted by a user."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_name = Column(String(500), nullable=False)
    video_path = Column(String(1000), nullable=False)
    status = Column(
        String(20),
        nullable=False,
        default="queued",
        index=True
    )  # queued, processing, phase1, phase2, completed, failed
    created_at = Column(DateTime, default=get_local_time)
    updated_at = Column(DateTime, default=get_local_time,
                        onupdate=get_local_time)
    error_message = Column(Text, nullable=True)

    # ROI coordinates (x1, y1, x2, y2) stored as JSON string
    roi_coords = Column(String(200), nullable=True)

    # Timestamp region coordinates for OCR crop (x1, y1, x2, y2) as JSON string
    timestamp_region_coords = Column(String(200), nullable=True)

    # Pipeline config snapshot stored as JSON
    config_json = Column(Text, nullable=True)
    agent = Column(String(50), nullable=True)

    # Progress info
    progress_message = Column(String(500), nullable=True)
    phase1_count = Column(Integer, default=0)
    phase2_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)

    # Timestamp info (from OCR of recording timestamp)
    recording_date = Column(String(50), nullable=True)   # e.g. "2026-01-25"
    recording_hour = Column(Integer, nullable=True)       # HH extracted from recording timestamp (0-23)

    # Output directory for this job's results
    output_dir = Column(String(1000), nullable=True)

    # Relationships
    results = relationship("Result", back_populates="job", cascade="all, delete-orphan")
    token_usages = relationship("TokenUsage", back_populates="job", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "video_name": self.video_name,
            "video_path": self.video_path,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error_message": self.error_message,
            "roi_coords": json.loads(self.roi_coords) if self.roi_coords else None,
            "timestamp_region_coords": json.loads(self.timestamp_region_coords) if self.timestamp_region_coords else None,
            "progress_message": self.progress_message,
            "phase1_count": self.phase1_count,
            "phase2_count": self.phase2_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "output_dir": self.output_dir,
            "recording_date": self.recording_date,
            "recording_hour": self.recording_hour,
            "agent": self.agent,
        }


class Result(Base):
    """Stores individual detection results for a completed job."""
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    event_id = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    scale = Column(String(20), nullable=True)
    phase1_reading = Column(Float, nullable=True)
    verified_reading = Column(Float, nullable=True)
    unit = Column(String(20), nullable=True)
    reading_state = Column(String(50), nullable=True)
    confidence = Column(Float, default=0.0)
    description = Column(Text, nullable=True)
    reading_correction = Column(String(200), nullable=True)
    real_timestamp = Column(String(20), nullable=True)  # HH:MM:SS wall-clock time

    # Relationship
    job = relationship("Job", back_populates="results")

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "event_id": self.event_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "scale": self.scale,
            "phase1_reading": self.phase1_reading,
            "verified_reading": self.verified_reading,
            "unit": self.unit,
            "reading_state": self.reading_state,
            "confidence": self.confidence,
            "description": self.description,
            "reading_correction": self.reading_correction,
            "real_timestamp": self.real_timestamp,
        }


class TokenUsage(Base):
    """Tracks token consumption per phase/batch."""
    __tablename__ = "token_usages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    phase = Column(String(20), nullable=False)   # "phase1", "phase2", "display_detection"
    model = Column(String(100), nullable=True)
    batch_or_clip_id = Column(Integer, nullable=True)
    # API-reported token counts
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cached_tokens = Column(Integer, default=0)
    # Estimated granular breakdown
    image_input_tokens = Column(Integer, default=0)
    text_input_tokens = Column(Integer, default=0)
    text_output_tokens = Column(Integer, default=0)
    # Cost (USD)
    input_cost_usd = Column(Float, default=0.0)
    output_cost_usd = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)

    # Relationship
    job = relationship("Job", back_populates="token_usages")


# =============================================================================
# Database Engine & Session Setup
# =============================================================================
from sqlalchemy import event as _sa_event

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})

# Enable WAL mode and tune SQLite for concurrent reader + writer access.
# WAL allows the polling reads (API) and the background worker writes to proceed
# without blocking each other, which is the main access pattern here.
@_sa_event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _connection_record):
    if DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-32000")   # 32 MB page cache
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get a database session (for use as a dependency or context manager)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# Backend Routing — transparently swap to Firebase when USE_FIREBASE=true
# =============================================================================
# After defining the SQLite engine above, we check the USE_FIREBASE flag.
# If enabled, we override init_db and SessionLocal so that the rest of the
# application (main.py, worker.py) continues to work without any changes.
# The Job/Result/TokenUsage names are also re-exported from firebase_db so
# that `from database import Job` works in both modes.

from config import USE_FIREBASE  # noqa: E402 — intentional late import

if USE_FIREBASE:
    import logging as _logging
    _logging.getLogger(__name__).info(
        "USE_FIREBASE=true — switching database backend to Firestore."
    )
    from firebase_db import (          # noqa: F401
        init_db,                       # overrides SQLite init_db above
        FirestoreSessionLocal as SessionLocal,  # overrides SQLite SessionLocal
        FirestoreJob as Job,           # overrides SQLAlchemy Job
        FirestoreResult as Result,     # overrides SQLAlchemy Result
        FirestoreTokenUsage as TokenUsage,  # overrides SQLAlchemy TokenUsage
    )
