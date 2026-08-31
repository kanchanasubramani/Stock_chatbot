"""
config.py
---------
Central place for environment variables and constants. Logging setup
lives in app/logger.py — a separate concern (see bottom of this file
for the re-export that keeps existing call sites working).
"""

import os

from pathlib import Path      # ← add this line

from dataclasses import dataclass


from dotenv import load_dotenv

from app.logger import get_logger  # noqa: F401  (re-exported for callers)

load_dotenv()  # reads .env into process environment (no-op if .env absent)

# ---------------------------------------------------------------------------
# API credentials / endpoints
# ---------------------------------------------------------------------------
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

ALPHA_VANTAGE_BASE_URL = os.getenv(
    "ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query"
)

# Cost-efficient default model; override via .env
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Cheapest current OpenAI embedding model — used only for RAG document
# retrieval (see rag_store.py), never for the chat completion itself.
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Cost / safety limits
# ---------------------------------------------------------------------------
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "200"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "200"))

# ---------------------------------------------------------------------------
# RAG (document retrieval) settings
# ---------------------------------------------------------------------------
# One folder per ticker under DOCUMENTS_DIR, e.g. data/documents/TSLA/*.pdf
BASE_DIR = Path(__file__).resolve().parent.parent  # .../stock-chatbot/

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "data" / "documents"))
EMBEDDINGS_CACHE_DIR = os.getenv("EMBEDDINGS_CACHE_DIR", str(BASE_DIR / "data" / "embeddings_cache"))

# Where cached chunk embeddings live (one JSON file per ticker). This cache
# is the main cost lever: a warm cache means ingest() re-embeds nothing.

# Target size of each chunk sent for embedding, and overlap between
# consecutive chunks (both in tokens, not characters).
RAG_CHUNK_SIZE_TOKENS = int(os.getenv("RAG_CHUNK_SIZE_TOKENS", "400"))
RAG_CHUNK_OVERLAP_TOKENS = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "50"))

# How many chunks retrieve() may return per question, at most.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# Hard ceiling (in tokens) on the total retrieved-document text injected
# into a single chatbot prompt, regardless of how many chunks scored well.
RAG_MAX_CONTEXT_TOKENS = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "800"))

# Cosine-similarity floor below which a chunk is considered irrelevant
# noise and dropped rather than spending prompt tokens on it.
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.15"))

# ---------------------------------------------------------------------------
# Conversation memory (rolling-window summarization)
# ---------------------------------------------------------------------------
# Number of most recent Q&A turns kept verbatim in the prompt. Anything
# older gets folded into a single running summary instead of being resent
# in full — see app/conversation.py.
CHAT_ROLLING_WINDOW_TURNS = int(os.getenv("CHAT_ROLLING_WINDOW_TURNS", "3"))

# ---------------------------------------------------------------------------
# SETTINGS — aggregate object consumed by guardrails.py
# ---------------------------------------------------------------------------
# guardrails.py was written against `from .config import SETTINGS` (an
# object with attribute access), while the rest of this file exposes plain
# module-level constants. This class bridges the two so guardrails.py can
# live inside the app/ package unmodified. Every field is still just an
# os.getenv() read, same as everything above — this is a grouping, not a
# new source of truth.
@dataclass(frozen=True)
class Settings:
    # PDF upload guardrails (validate_pdf_upload / validate_pdf_page_count)
    max_pdf_files: int = int(os.getenv("MAX_PDF_FILES", "5"))
    max_pdf_mb: float = float(os.getenv("MAX_PDF_MB", "20"))
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "300"))

    # Question length / rate limit / session budget
    question_max_chars: int = int(os.getenv("QUESTION_MAX_CHARS", str(MAX_INPUT_CHARS)))
    min_seconds_between_requests: float = float(os.getenv("MIN_SECONDS_BETWEEN_REQUESTS", "2"))
    max_questions_per_session: int = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "50"))

    # Content moderation
    enable_moderation: bool = os.getenv("ENABLE_MODERATION", "true").lower() == "true"
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))

    # Legacy hard-truncation chat history cutoff (guardrails.trim_history).
    # app.py uses conversation.py's rolling-summary approach instead, but
    # trim_history() is kept available for simpler callers.
    max_chat_turns: int = int(os.getenv("MAX_CHAT_TURNS", "10"))

    # RAG folder-level ingestion limits (guardrails.validate_documents_folder)
    rag_max_documents_per_symbol: int = int(os.getenv("RAG_MAX_DOCUMENTS_PER_SYMBOL", "10"))
    rag_max_total_mb_per_symbol: float = float(os.getenv("RAG_MAX_TOTAL_MB_PER_SYMBOL", "100"))


SETTINGS = Settings()

# ---------------------------------------------------------------------------
# Technical indicator periods
# ---------------------------------------------------------------------------
SMA_SHORT_PERIOD = 20
SMA_MEDIUM_PERIOD = 50
SMA_LONG_PERIOD = 200

RSI_PERIOD = 14

MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# ---------------------------------------------------------------------------
# Logging note: setup + get_logger() live in app/logger.py, imported above.
# ---------------------------------------------------------------------------


def validate_config() -> list[str]:
    """Return a list of missing required config values (empty list = ok).
    Called once at app startup so failures are surfaced immediately with a
    clear message instead of a confusing downstream error."""
    missing = []
    if not ALPHA_VANTAGE_API_KEY:
        missing.append("ALPHA_VANTAGE_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    return missing
