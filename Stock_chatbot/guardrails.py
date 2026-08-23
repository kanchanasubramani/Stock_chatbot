"""
Guardrails for the stock analysis chatbot.

Covers:
- Input validation (ticker format, PDF size/pages/count, question length)
- Prompt-injection defense (delimiter-wrapping + sanitizing untrusted text
  pulled from PDFs or API responses, since that text is fed to the LLM but
  was never written by the operator and could contain embedded instructions)
- Content moderation (OpenAI Moderation API) on user input, fail-open if the
  moderation call itself errors so a moderation outage doesn't take the app down
- Rate limiting / per-session cost budget
- Output guardrail (always carries the "not financial advice" disclaimer)
- Chat history trimming to bound token cost
"""
import re
import time
from typing import Optional

from openai import OpenAI

from .config import SETTINGS
from .logger import get_logger

logger = get_logger(__name__)

TICKER_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,3})?$")

DISCLAIMER = (
    "\n\n---\n*This is research/educational information, not financial advice. "
    "Verify independently and consult a licensed financial advisor before making "
    "investment decisions.*"
)


class GuardrailError(Exception):
    """User-facing guardrail violation. The message is safe to display as-is."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_ticker(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise GuardrailError("Please enter a ticker symbol.")
    if not TICKER_RE.match(symbol):
        raise GuardrailError(
            f"'{symbol}' doesn't look like a valid ticker symbol "
            "(letters, optional '.' suffix, e.g. AAPL, BRK.B)."
        )
    return symbol


def validate_pdf_upload(file, existing_count: int) -> None:
    if existing_count >= SETTINGS.max_pdf_files:
        raise GuardrailError(
            f"Maximum of {SETTINGS.max_pdf_files} uploaded PDFs per session. "
            "Clear documents to upload different ones."
        )
    size_mb = file.size / (1024 * 1024)
    if size_mb > SETTINGS.max_pdf_mb:
        raise GuardrailError(
            f"'{file.name}' is {size_mb:.1f} MB, which exceeds the {SETTINGS.max_pdf_mb:.0f} MB limit."
        )


def validate_pdf_page_count(num_pages: int, filename: str) -> None:
    if num_pages > SETTINGS.max_pdf_pages:
        raise GuardrailError(
            f"'{filename}' has {num_pages} pages, which exceeds the "
            f"{SETTINGS.max_pdf_pages}-page limit."
        )


def validate_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise GuardrailError("Please enter a question.")
    if len(question) > SETTINGS.question_max_chars:
        raise GuardrailError(
            f"Please keep questions under {SETTINGS.question_max_chars} characters."
        )
    return question


# ---------------------------------------------------------------------------
# Rate limiting / session budget (in-memory, per Streamlit session)
# ---------------------------------------------------------------------------
def check_rate_limit(last_request_ts: Optional[float]) -> None:
    if last_request_ts is None:
        return
    elapsed = time.time() - last_request_ts
    if elapsed < SETTINGS.min_seconds_between_requests:
        wait = SETTINGS.min_seconds_between_requests - elapsed
        raise GuardrailError(f"Please wait {wait:.1f}s before sending another question.")


def check_session_budget(question_count: int) -> None:
    if question_count >= SETTINGS.max_questions_per_session:
        raise GuardrailError(
            f"This session has reached its limit of {SETTINGS.max_questions_per_session} "
            "questions. Refresh the app to start a new session."
        )


# ---------------------------------------------------------------------------
# Prompt-injection defense
# ---------------------------------------------------------------------------
def sanitize_untrusted_text(text: str) -> str:
    """Strip sequences commonly used to break out of prompt delimiters."""
    return text.replace("```", "'''").replace("<<", "‹‹").replace(">>", "››")


def wrap_untrusted_context(label: str, text: str) -> str:
    """
    Wrap externally-sourced text (PDF content, API responses) in explicit,
    labeled delimiters so the system prompt can instruct the model to treat
    everything inside as inert reference data, never as instructions —
    this is the core defense against prompt injection hidden in a PDF or
    in unusual ticker/API-field values.
    """
    safe_text = sanitize_untrusted_text(text)
    return f"<{label}_untrusted_data>\n{safe_text}\n</{label}_untrusted_data>"


# ---------------------------------------------------------------------------
# Content moderation (OpenAI Moderation API) — defense in depth
# ---------------------------------------------------------------------------
def moderate_text(api_key: str, text: str) -> None:
    """Raise GuardrailError if text trips OpenAI's moderation categories.
    Fails open (logs + allows through) if the moderation call itself errors,
    so a moderation-endpoint outage doesn't take the whole app down.
    """
    if not SETTINGS.enable_moderation or not text.strip():
        return
    try:
        client = OpenAI(api_key=api_key, timeout=SETTINGS.request_timeout_seconds)
        result = client.moderations.create(model="omni-moderation-latest", input=text)
        flagged = result.results[0].flagged
        if flagged:
            cats = result.results[0].categories.model_dump()
            flagged_cats = [c for c, v in cats.items() if v]
            logger.warning("Moderation flagged content: %s", flagged_cats)
            raise GuardrailError(
                "This message was flagged by content moderation and can't be processed. "
                "Please rephrase your question."
            )
    except GuardrailError:
        raise
    except Exception as e:
        logger.error("Moderation check failed, failing open: %s", e)


# ---------------------------------------------------------------------------
# Output guardrail
# ---------------------------------------------------------------------------
def ensure_disclaimer(answer: str) -> str:
    if "not financial advice" not in answer.lower():
        answer = answer.rstrip() + DISCLAIMER
    return answer


# ---------------------------------------------------------------------------
# Chat history bounding (cost control)
# ---------------------------------------------------------------------------
def trim_history(history: list) -> list:
    """Keep only the most recent N turns sent to the LLM, to bound token cost.
    (The full history can still be shown in the UI independently.)
    """
    max_messages = SETTINGS.max_chat_turns * 2  # each turn = user + assistant message
    if len(history) > max_messages:
        return history[-max_messages:]
    return history
