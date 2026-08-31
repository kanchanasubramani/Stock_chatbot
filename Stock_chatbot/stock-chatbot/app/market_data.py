"""
market_data.py
---------------
Talks to Alpha Vantage's TIME_SERIES_DAILY endpoint and hands back raw
JSON time-series data. This is the ONLY module that knows about the
Alpha Vantage HTTP contract for price data.
"""

import requests

from app import config

logger = config.get_logger(__name__)


class MarketDataError(Exception):
    """Base exception for all market data failures."""


class InvalidTickerError(MarketDataError):
    """Raised when Alpha Vantage reports the symbol doesn't exist."""


class RateLimitError(MarketDataError):
    """Raised when Alpha Vantage's call frequency limit is hit."""


class MarketData:
    """Fetches daily OHLCV data for a single ticker."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or config.ALPHA_VANTAGE_API_KEY
        self.base_url = base_url or config.ALPHA_VANTAGE_BASE_URL

    def fetch_daily(self, symbol: str, outputsize: str = "compact") -> dict:
        """
        Fetch daily OHLCV data for `symbol`.

        Returns the raw "Time Series (Daily)" dict from Alpha Vantage:
            { "2024-06-10": {"1. open": "...", ...}, ... }

        Raises InvalidTickerError, RateLimitError, or MarketDataError.
        """
        symbol = symbol.strip().upper()
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }

        logger.info("API request started | endpoint=TIME_SERIES_DAILY | symbol=%s", symbol)

        try:
            response = requests.get(
                self.base_url, params=params, timeout=config.SETTINGS.request_timeout_seconds
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            logger.error("API request failed | symbol=%s | reason=timeout", symbol)
            raise MarketDataError(f"Request to Alpha Vantage timed out for {symbol}") from exc
        except requests.exceptions.RequestException as exc:
            logger.error("API request failed | symbol=%s | reason=network_error", symbol)
            raise MarketDataError(f"Network error fetching data for {symbol}") from exc

        data = response.json()

        # Alpha Vantage returns HTTP 200 even for errors — it signals them via keys.
        if "Error Message" in data:
            logger.warning("Invalid ticker | symbol=%s", symbol)
            raise InvalidTickerError(f"'{symbol}' is not a valid ticker symbol.")

        if "Note" in data or "Information" in data:
            logger.warning("Rate limit | symbol=%s", symbol)
            raise RateLimitError(
                "Alpha Vantage rate limit reached. Please wait a moment and try again."
            )

        series = data.get("Time Series (Daily)")
        if not series:
            logger.error("API request failed | symbol=%s | reason=no_data", symbol)
            raise MarketDataError(f"No price data returned for {symbol}.")

        logger.info("API request succeeded | symbol=%s | rows=%d", symbol, len(series))
        return series