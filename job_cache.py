"""
In-memory write-through cache for job metadata.

Eliminates repeated Firestore list/poll reads — the primary source of quota
exhaustion. Both main.py and worker.py write through this cache on every
mutation, so the in-memory view stays current without any extra reads.

Usage
-----
  import job_cache

  # Startup — load all existing jobs once
  job_cache.populate(list_of_job_dicts)

  # Reads — zero Firestore calls
  job_cache.get_all()          # sorted list, newest first
  job_cache.get(job_id)        # single job dict or None

  # Writes — called automatically by firebase_db commit / delete
  job_cache.upsert(job_dict)   # insert or update; signals worker if queued
  job_cache.remove(job_id)     # evict job + its cached results

  # Results caching (optional, per-job)
  job_cache.set_results(job_id, result_list)
  job_cache.get_results(job_id)       # None if not yet cached
  job_cache.invalidate_results(job_id)

  # Worker coordination — replaces sleep(3) + Firestore poll
  job_cache.wait_for_queued(timeout)  # blocks until a queued job appears
"""
import threading
from typing import Optional

_lock = threading.RLock()
_jobs: dict = {}      # str(job_id) -> job_dict (from to_dict())
_results: dict = {}   # str(job_id) -> list[result_dict]
_initialized: bool = False
_queued_event = threading.Event()   # set when a job enters "queued" status


# ── Population ───────────────────────────────────────────────────────────────

def populate(job_dicts: list):
    """Load all existing jobs into the cache (called once at startup)."""
    global _initialized
    with _lock:
        for j in job_dicts:
            _jobs[str(j["id"])] = j
        _initialized = True
    # Signal worker in case any loaded jobs are already queued
    if any(j.get("status") == "queued" for j in job_dicts):
        _queued_event.set()


# ── Reads ────────────────────────────────────────────────────────────────────

def is_ready() -> bool:
    """Return True once populate() has been called."""
    return _initialized


def get_all() -> Optional[list]:
    """
    Return all jobs sorted by created_at descending (newest first), capped at 100.
    Returns None if the cache has not been populated yet.
    """
    if not _initialized:
        return None
    with _lock:
        jobs = list(_jobs.values())
    jobs.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return jobs[:100]


def get(job_id) -> Optional[dict]:
    """Return a single job dict, or None if not in cache."""
    with _lock:
        return _jobs.get(str(job_id))


# ── Writes ───────────────────────────────────────────────────────────────────

def upsert(job_dict: dict):
    """Insert or replace a job in the cache. Signals worker if status=queued."""
    with _lock:
        _jobs[str(job_dict["id"])] = job_dict
    if job_dict.get("status") == "queued":
        _queued_event.set()


def remove(job_id):
    """Evict a job and its cached results from the cache."""
    with _lock:
        _jobs.pop(str(job_id), None)
        _results.pop(str(job_id), None)


# ── Results caching ──────────────────────────────────────────────────────────

def set_results(job_id, result_list: list):
    with _lock:
        _results[str(job_id)] = result_list


def get_results(job_id) -> Optional[list]:
    with _lock:
        return _results.get(str(job_id))


def invalidate_results(job_id):
    with _lock:
        _results.pop(str(job_id), None)


# ── Worker coordination ──────────────────────────────────────────────────────

def wait_for_queued(timeout: float = 30.0) -> bool:
    """
    Block until a job enters 'queued' status or timeout elapses.
    Returns True if the event was signalled (a queued job probably exists),
    False on timeout.
    """
    triggered = _queued_event.wait(timeout=timeout)
    _queued_event.clear()
    return triggered
