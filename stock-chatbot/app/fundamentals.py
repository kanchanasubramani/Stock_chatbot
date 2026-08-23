"""
fundamentals.py
----------------
Fetches company fundamentals from Alpha Vantage's OVERVIEW endpoint and
extracts a compact, safe subset of fields. Missing fields are handled
gracefully (returned as None) rather than raising.
"""

import requests

from app import config

logger = config.get_logger(__name__)

# Alpha Vantage field name -> our compact field name
_FIELD_MAP = {
    "Name": "name",
    "Sector": "sector",
    "Industry": "industry",
    "MarketCapitalization": "market_cap",
    "PERatio": "pe",
    "PEGRatio": "peg",
    "PriceToBookRatio": "price_to_book",
    "EPS": "eps",
    "ProfitMargin": "profit_margin",
    "OperatingMarginTTM": "operating_margin",
    "ReturnOnEquityTTM": "roe",
    "RevenueTTM": "revenue",
    "QuarterlyRevenueGrowthYOY": "revenue_growth",
    "QuarterlyEarningsGrowthYOY": "earnings_growth",
}

_NUMERIC_FIELDS = {
    "market_cap", "pe", "peg", "price_to_book", "eps", "profit_margin",
    "operating_margin", "roe", "revenue", "revenue_growth", "earnings_growth",
}


class FundamentalsError(Exception):
    pass


class Fundamentals:
    """Fetches and extracts compact fundamental metrics for a ticker."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or config.ALPHA_VANTAGE_API_KEY
        self.base_url = base_url or config.ALPHA_VANTAGE_BASE_URL

    def fetch(self, symbol: str) -> dict:
        symbol = symbol.strip().upper()
        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": self.api_key}

        logger.info("API request started | endpoint=OVERVIEW | symbol=%s", symbol)

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("API request failed | symbol=%s | reason=network_error", symbol)
            raise FundamentalsError(f"Network error fetching fundamentals for {symbol}") from exc

        data = response.json()

        if "Note" in data or "Information" in data:
            logger.warning("Rate limit | symbol=%s", symbol)
            raise FundamentalsError("Alpha Vantage rate limit reached for fundamentals.")

        if not data or "Symbol" not in data:
            # OVERVIEW returns {} for tickers with no fundamental data (e.g. some ETFs).
            logger.warning("No fundamentals available | symbol=%s", symbol)
            return {v: None for v in _FIELD_MAP.values()}

        result = {}
        for av_key, our_key in _FIELD_MAP.items():
            raw_value = data.get(av_key)
            result[our_key] = _safe_value(raw_value, is_numeric=our_key in _NUMERIC_FIELDS)

        logger.info("API request succeeded | endpoint=OVERVIEW | symbol=%s", symbol)
        return result


def _safe_value(raw, is_numeric: bool):
    """Alpha Vantage uses the literal string 'None' for missing fields."""
    if raw is None or raw == "None" or raw == "":
        return None
    if is_numeric:
        try:
            return round(float(raw), 4)
        except (TypeError, ValueError):
            return None
    return raw
