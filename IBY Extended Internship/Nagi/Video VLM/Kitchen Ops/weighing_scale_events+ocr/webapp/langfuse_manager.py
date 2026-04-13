"""
Langfuse Manager — Centralized client for prompt management and tracing.

SDK version: Langfuse v3.x API:
  - Root trace:    lf.start_observation(name, type="SPAN", ...).__enter__()
  - Child span:    lf.start_observation(name, type="SPAN", ...).__enter__()  (auto-nested via contextvar)
  - Update span:   lf.update_current_span(metadata=...)
  - Close span:    ctx.__exit__(None, None, None)
  - Flush:         lf.flush()

Usage:
    from langfuse_manager import get_langfuse, get_prompt, start_trace, start_child_span, end_span, flush
"""
import logging
from typing import Optional, Dict, Any, List

import config as app_config

logger = logging.getLogger(__name__)

# ── Singleton ────────────────────────────────────────────────────────────────

_langfuse_client = None


def get_langfuse():
    """
    Return a singleton Langfuse client.
    Initializes lazily on first call. Returns None if keys are missing.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    public_key = app_config.LANGFUSE_PUBLIC_KEY
    secret_key = app_config.LANGFUSE_SECRET_KEY
    host = app_config.LANGFUSE_HOST

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse keys not set (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Tracing and prompt management are DISABLED."
        )
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info(f"Langfuse client initialized (host={host})")
        return _langfuse_client
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        return None


# ── Prompt Management ────────────────────────────────────────────────────────

def get_prompt(
    prompt_name: str,
    fallback: str = "",
    variables: Optional[Dict[str, str]] = None,
    label: str = "production",
) -> str:
    """
    Fetch a prompt from Langfuse by name.

    The prompts are stored in Langfuse with Python {var} style placeholders
    (matching how the agent code uses them). We fetch the raw text from
    Langfuse and apply variables via str.format().

    If Langfuse is unavailable or the prompt doesn't exist, returns *fallback*.

    Args:
        prompt_name: Name registered in Langfuse (e.g. "pork-weighing-phase1").
        fallback:    Hardcoded fallback prompt string (original agent prompt).
        variables:   Template variables applied via str.format() (e.g. {"context": "..."}).
        label:       Langfuse prompt label to target (default "production").

    Returns:
        Compiled prompt string.
    """
    lf = get_langfuse()

    if lf is None:
        logger.debug(f"Langfuse unavailable — using local fallback for '{prompt_name}'")
        return _apply_variables(fallback, variables)

    try:
        prompt_obj = lf.get_prompt(prompt_name, label=label, type="text")
        # get_prompt returns a TextPromptClient; .compile() applies Langfuse {{var}} vars.
        # Our prompts use Python {var} syntax so we call compile() with no args,
        # then apply variables ourselves with str.format().
        raw_text = prompt_obj.compile()
        compiled = _apply_variables(raw_text, variables)
        logger.info(f"Loaded prompt '{prompt_name}' v{prompt_obj.version} from Langfuse")
        return compiled
    except Exception as e:
        logger.warning(
            f"Could not fetch prompt '{prompt_name}' from Langfuse ({e}). "
            f"Using local fallback."
        )
        return _apply_variables(fallback, variables)


def _apply_variables(template: str, variables: Optional[Dict[str, str]]) -> str:
    """Apply a variables dict to a str.format()-style template, gracefully."""
    if not variables or not template:
        return template
    try:
        return template.format(**variables)
    except (KeyError, IndexError):
        return template


# ── Tracing Helpers (Langfuse v3 SDK) ───────────────────────────────────────
#
# v3 API uses start_observation() + context vars:
#   lf.start_observation(name, type, metadata, tags, session_id) → context manager
#   Entering the context makes it the "current" observation for nesting.
#   Child observations automatically attach to the active parent via contextvar.
#   lf.update_current_span(metadata=...) updates the active observation.

def start_trace(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,  # noqa: ARG001 — kept for API compatibility
    session_id: Optional[str] = None,  # noqa: ARG001 — kept for API compatibility
):
    """
    Start a root-level Langfuse span.
    Returns the span object, or None if Langfuse is unavailable.
    Caller must call end_span() when done.
    """
    lf = get_langfuse()
    if lf is None:
        return None

    try:
        return lf.start_observation(
            name=name,
            as_type="span",
            metadata=metadata or {},
        )
    except Exception as e:
        logger.warning(f"Failed to create Langfuse root span '{name}': {e}")
        return None


def start_child_span(parent_span, name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Start a child span nested under parent_span.
    """
    if parent_span is None:
        return None
    try:
        return parent_span.start_observation(
            name=name,
            as_type="span",
            metadata=metadata or {},
        )
    except Exception as e:
        logger.warning(f"Failed to create child span '{name}': {e}")
        return None


def end_span(span, metadata: Optional[Dict[str, Any]] = None, output: Optional[Any] = None):
    """Safely end a span, optionally attaching output and metadata."""
    if span is None:
        return
    try:
        update_kwargs: Dict[str, Any] = {}
        if metadata:
            update_kwargs["metadata"] = metadata
        if output is not None:
            update_kwargs["output"] = output
        if update_kwargs:
            try:
                span.update(**update_kwargs)
            except Exception:
                pass
        span.end()
    except Exception as e:
        logger.warning(f"Failed to end Langfuse span: {e}")


def log_generation(
    parent,
    name: str,
    model: str,
    usage_input: int,
    usage_output: int,
    usage_total: int,
    cost_usd: float,
    input_cost_usd: float = 0.0,
    output_cost_usd: float = 0.0,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    usage_details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an LLM generation to Langfuse using the 'generation' observation type.

    This populates Langfuse's **Tokens** and **Total Cost** columns — generic
    spans with token counts in metadata do not surface there.

    Args:
        parent:          Root or parent span returned by start_trace / start_child_span.
        name:            Generation label shown in the Langfuse trace tree.
        model:           Model ID string (e.g. "gpt-5-mini", "gemini-3-flash-preview").
        usage_input:     Total prompt/input tokens (text + images combined).
        usage_output:    Completion/output tokens.
        usage_total:     Sum of input + output (may include cached tokens).
        cost_usd:        Total call cost in USD.
        input_cost_usd:  Input portion of cost (used for Langfuse cost split).
        output_cost_usd: Output portion of cost.
        input_text:      Optional prompt text to attach as input.
        output_text:     Optional response text to attach as output.
        metadata:        Arbitrary key-value metadata (e.g. batch_id, frame count).
        usage_details:   Granular token breakdown stored under usage_details
                         (e.g. image_input_tokens, text_input_tokens, cached_tokens).
    """
    if parent is None:
        return
    lf = get_langfuse()
    if lf is None:
        return
    try:
        # Langfuse v3 API: token counts go in usage_details (Dict[str, int]),
        # costs go in cost_details (Dict[str, float]).
        # The old `usage={...}` dict is NOT a valid parameter in v3.x.
        _usage: Dict[str, int] = {
            "input":  int(usage_input),
            "output": int(usage_output),
            "total":  int(usage_total),
        }
        if usage_details:
            # Merge extra breakdown (image_input_tokens etc.) — must be ints
            _usage.update({k: int(v) for k, v in usage_details.items()})

        gen = parent.start_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input_text,
            output=output_text,
            usage_details=_usage,
            cost_details={
                "input":  float(input_cost_usd),
                "output": float(output_cost_usd),
                "total":  float(cost_usd),
            },
            metadata=metadata or {},
        )
        gen.end()
    except Exception as e:
        logger.warning(f"log_generation '{name}' failed (non-fatal): {e}")


def flush():
    """Flush any pending Langfuse events (call at shutdown or end of a job)."""
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass
