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
import os
import re
import time
from typing import Optional

import tiktoken
from openai import OpenAI
"""OpenAI Moderation API:
 A free service for developers that analyzes text and images 
 using multimodal models like omni-moderation-latest."""

from .config import SETTINGS
from .logger import get_logger

logger = get_logger(__name__)

# Loaded lazily so importing this module never requires network access.
_encoding = None
"""
cl100k_base is an open-source byte-pair encoding (BPE) 
tokenizer created by OpenAI for models like GPT-4, GPT-3.5-Turbo, and text-embedding-ada-002."""

def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding
"""What tokenizer encodings are used for

An encoding defines how raw text gets split into tokens — the numeric units a language model actually processes. They matter for:

Cost and context limits –
    API pricing and context windows are measured in tokens,
    not characters or words, so the encoding determines
    ow much text "fits" and what you're billed for.

Model training/inference – the model was
    trained on sequences of these specific token IDs, 
so encoding must match exactly at inference time.


Efficiency – better encodings pack more meaning per 
token (especially for code, non-English languages, 
whitespace-heavy text like Python), which lowers 
cost and effectively extends usable context.

"""

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


def validate_documents_folder(symbol: str, docs_dir: str) -> list[str]:
    """
    Bound the cost of RAG ingestion at the folder level, before any PDF is
    even opened: caps how many files and how much total disk size a single
    ticker's document folder may contain. Limits come from
    SETTINGS.rag_max_documents_per_symbol / SETTINGS.rag_max_total_mb_per_symbol.

    Returns the list of PDF filenames accepted for ingestion.
    """
    if not os.path.isdir(docs_dir):
        return []

    max_files = SETTINGS.rag_max_documents_per_symbol
    max_total_mb = SETTINGS.rag_max_total_mb_per_symbol

    pdf_files = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf"))
    if len(pdf_files) > max_files:
        logger.warning(
            "RAG folder exceeds file limit | symbol=%s | found=%d | limit=%d",
            symbol, len(pdf_files), max_files,
        )
        pdf_files = pdf_files[:max_files]

    accepted, total_mb = [], 0.0
    for filename in pdf_files:
        size_mb = os.path.getsize(os.path.join(docs_dir, filename)) / (1024 * 1024)
        if total_mb + size_mb > max_total_mb:
            logger.warning(
                "RAG folder exceeds total size limit | symbol=%s | limit_mb=%.1f",
                symbol, max_total_mb,
            )
            break
        accepted.append(filename)
        total_mb += size_mb

    return accepted


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
# RAG document guardrails
# ---------------------------------------------------------------------------
# Retrieved PDF chunks are untrusted the same way an uploaded PDF or a raw
# API field is: none of that text was written by the operator, and a
# malicious or poisoned document could contain embedded instructions. These
# helpers reuse the same sanitize/wrap primitives above rather than
# introducing a second defense path.

def sanitize_rag_chunks(chunks: list[dict]) -> list[dict]:
    """Apply the same delimiter-breakout sanitization used for PDFs/API
    fields to every retrieved chunk's text, in place of the raw text."""
    return [{**chunk, "text": sanitize_untrusted_text(chunk["text"])} for chunk in chunks]


def enforce_rag_token_budget(chunks: list[dict], max_tokens: int) -> list[dict]:
    """Defense-in-depth token cap. rag_store.retrieve() already budgets its
    own output, but this re-checks at the guardrails boundary so a bug or a
    future direct call into DocumentStore can't silently blow past the
    intended prompt-token budget."""
    encoding = _get_encoding()
    kept, used = [], 0
    for chunk in chunks:
        tokens = len(encoding.encode(chunk["text"]))
        if used + tokens > max_tokens:
            break
        kept.append(chunk)
        used += tokens
    return kept


def wrap_rag_context(chunks: list[dict], max_tokens: Optional[int] = None) -> str:
    """
    Sanitize, token-budget, and format retrieved document chunks into a
    single labeled block ready to drop into a chatbot prompt, with source
    citations so the model (and the guardrail-checked answer) can point
    back to where a claim came from.

    Returns "" if there are no chunks — callers should skip adding a
    Documents: section to the prompt entirely in that case, to avoid
    spending tokens on an empty block.
    """
    if not chunks:
        return ""

    clean_chunks = sanitize_rag_chunks(chunks)
    if max_tokens is not None:
        clean_chunks = enforce_rag_token_budget(clean_chunks, max_tokens)

    parts = [
        f"[Source: {c['source_file']}, page {c['page']}]\n{c['text']}"
        for c in clean_chunks
    ]
    return wrap_untrusted_context("document", "\n\n".join(parts))


def wrap_conversation_context(context_block: str) -> str:
    """
    Sanitize and delimiter-wrap prior-conversation text (rolling summary +
    recent verbatim turns, built by conversation.ConversationManager) before
    it goes into a chatbot prompt. Past turns include the user's own earlier
    messages, which are untrusted the same way the current question is —
    wrapping them closes off a multi-turn prompt-injection path where an
    early message tries to plant instructions for the model to "remember"
    and follow in a later turn.

    Returns "" if there's no conversation yet, so callers can skip adding a
    History: section entirely.
    """
    if not context_block:
        return ""
    return wrap_untrusted_context("conversation", sanitize_untrusted_text(context_block))


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
