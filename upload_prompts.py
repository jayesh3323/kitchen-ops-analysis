"""
Upload all agent prompts to Langfuse Prompt Registry.

Run this script ONCE (or whenever prompts are updated) to push
the latest prompt text to Langfuse so agents can fetch them at runtime.

Usage:
    cd webapp
    python upload_prompts.py
"""
import sys
import os
import logging
from pathlib import Path

# Ensure webapp and parent directories are on sys.path
_webapp_dir = Path(__file__).parent
_parent_dir = _webapp_dir.parent
for p in [str(_webapp_dir), str(_parent_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config as app_config  # noqa: E402 — loads .env
from langfuse_manager import get_langfuse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Collect prompts from each agent ──────────────────────────────────────────

def _load_pork_weighing_prompts():
    """Load prompts from pork_weighing_compliance.py."""
    from agents.pork_weighing_compliance import PHASE1_SYSTEM_PROMPT, PHASE2_SYSTEM_PROMPT
    return {
        "pork-weighing-phase1": PHASE1_SYSTEM_PROMPT,
        "pork-weighing-phase2": PHASE2_SYSTEM_PROMPT,
    }


def _load_plating_time_prompts():
    """Load prompts from plating_time.py."""
    from agents.plating_time import PHASE1_SYSTEM_PROMPT, PHASE2_SYSTEM_PROMPT
    return {
        "plating-time-phase1": PHASE1_SYSTEM_PROMPT,
        "plating-time-phase2": PHASE2_SYSTEM_PROMPT,
    }


def _load_serve_time_prompts():
    """Load prompts from avg_serve_time.py."""
    from agents.avg_serve_time import PHASE1_SYSTEM_PROMPT, PHASE2_SYSTEM_PROMPT
    return {
        "serve-time-phase1": PHASE1_SYSTEM_PROMPT,
        "serve-time-phase2": PHASE2_SYSTEM_PROMPT,
    }


def _load_bowl_completion_prompts():
    """Load prompts from bowl_completion_rate.py."""
    from agents.bowl_completion_rate import PHASE1_SYSTEM_PROMPT, PHASE2_SYSTEM_PROMPT
    return {
        "bowl-completion-phase1": PHASE1_SYSTEM_PROMPT,
        "bowl-completion-phase2": PHASE2_SYSTEM_PROMPT,
    }


def _load_noodle_rotation_prompts():
    """Load prompts from noodle_rotation_compliance.py."""
    from agents.noodle_rotation_compliance import PHASE1_SYSTEM_PROMPT, PHASE2_SYSTEM_PROMPT
    return {
        "noodle-rotation-phase1": PHASE1_SYSTEM_PROMPT,
        "noodle-rotation-phase2": PHASE2_SYSTEM_PROMPT,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    lf = get_langfuse()
    if lf is None:
        logger.error(
            "Cannot upload prompts — Langfuse client failed to initialize. "
            "Check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env"
        )
        sys.exit(1)

    # Gather all prompts
    all_prompts = {}

    logger.info("Loading Pork Weighing prompts...")
    all_prompts.update(_load_pork_weighing_prompts())

    logger.info("Loading Plating Time prompts...")
    all_prompts.update(_load_plating_time_prompts())

    logger.info("Loading Serve Time prompts...")
    all_prompts.update(_load_serve_time_prompts())

    logger.info("Loading Bowl Completion Rate prompts...")
    all_prompts.update(_load_bowl_completion_prompts())

    logger.info("Loading Noodle Rotation prompts...")
    all_prompts.update(_load_noodle_rotation_prompts())

    logger.info(f"Collected {len(all_prompts)} prompts to upload.\n")

    # Upload each prompt
    success_count = 0
    for prompt_name, prompt_text in all_prompts.items():
        try:
            lf.create_prompt(
                name=prompt_name,
                prompt=prompt_text,
                labels=["production", "latest"],
                type="text",
            )
            char_count = len(prompt_text)
            logger.info(f"  ✅  {prompt_name}  ({char_count:,} chars)")
            success_count += 1
        except Exception as e:
            logger.error(f"  ❌  {prompt_name}  — {e}")

    # Flush to ensure all events are sent
    lf.flush()

    print(f"\nDone. {success_count}/{len(all_prompts)} prompts uploaded to Langfuse.")
    if success_count == len(all_prompts):
        print("All prompts are now available in the Langfuse Prompt Registry.")
        print("Agents will fetch them at runtime via langfuse_manager.get_prompt().")
    else:
        print("Some prompts failed. Check the errors above.")


if __name__ == "__main__":
    main()
