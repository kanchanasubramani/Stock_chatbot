"""
data_processor.py
------------------
Converts the raw Alpha Vantage time-series JSON into a clean, typed,
chronologically-sorted Pandas DataFrame. Nothing in this module talks
to the network or the LLM — pure transformation only.
"""

import pandas as pd

from app import config

logger = config.get_logger(__name__)

_COLUMN_MAP = {
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume",
}


class DataProcessor:
    """Turns raw Alpha Vantage JSON into a usable DataFrame."""

    @staticmethod
    def to_dataframe(raw_series: dict) -> pd.DataFrame:
        """
        raw_series: { "2024-06-10": {"1. open": "...", ...}, ... }
        Returns a DataFrame indexed by DatetimeIndex, sorted ascending,
        with columns Open, High, Low, Close, Volume (all float).
        """
        df = pd.DataFrame.from_dict(raw_series, orient="index")
        df = df.rename(columns=_COLUMN_MAP)
        df = df[list(_COLUMN_MAP.values())]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.dropna()

        logger.info("Processed DataFrame | rows=%d | range=%s to %s",
                    len(df), df.index.min().date() if len(df) else "n/a",
                    df.index.max().date() if len(df) else "n/a")
        return df
