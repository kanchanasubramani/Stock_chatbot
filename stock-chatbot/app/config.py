"""
config.py
---------
Central place for environment variables and constants. Logging setup
lives in app/logger.py — a separate concern (see bottom of this file
for the re-export that keeps existing call sites working).
"""

import os

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

# ---------------------------------------------------------------------------
# Cost / safety limits
# ---------------------------------------------------------------------------
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "200"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "200"))

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
