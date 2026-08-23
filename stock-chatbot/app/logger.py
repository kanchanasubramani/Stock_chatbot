"""
logger.py
---------
All logging setup lives here, separate from config.py, so logging
concerns (format, handlers, level) are isolated from application
settings (API keys, periods, limits). Every other module gets its
logger via get_logger(__name__) from here.
"""

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the app's standard format.
    Use this instead of logging.getLogger directly so every module is
    consistent and easy to grep for later."""
    return logging.getLogger(name)
