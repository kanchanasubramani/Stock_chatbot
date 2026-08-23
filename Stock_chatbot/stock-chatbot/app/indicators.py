"""
indicators.py
--------------
Deterministic technical-indicator calculations. All math lives here —
the LLM never sees a price series, only the final numbers this module
produces (see analysis.py / stock_analyzer.py for how they get trimmed
down further before being sent anywhere).
"""

import pandas as pd

from app import config

logger = config.get_logger(__name__)


class Indicators:
    """Calculates SMA, EMA, RSI, and MACD from a price DataFrame."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # neutral default when undefined (e.g. no losses)

    @staticmethod
    def macd(series: pd.Series):
        """Returns (macd_line, signal_line) as Series."""
        ema_fast = Indicators.ema(series, config.MACD_FAST_PERIOD)
        ema_slow = Indicators.ema(series, config.MACD_SLOW_PERIOD)
        macd_line = ema_fast - ema_slow
        signal_line = Indicators.ema(macd_line, config.MACD_SIGNAL_PERIOD)
        return macd_line, signal_line

    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> dict:
        """
        Compute every indicator and return only the LATEST value of each —
        this is the compact form that flows downstream toward the LLM.
        """
        close = df["Close"]

        sma20 = cls.sma(close, config.SMA_SHORT_PERIOD)
        sma50 = cls.sma(close, config.SMA_MEDIUM_PERIOD)
        sma200 = cls.sma(close, config.SMA_LONG_PERIOD)
        ema20 = cls.ema(close, config.SMA_SHORT_PERIOD)
        rsi14 = cls.rsi(close)
        macd_line, macd_signal = cls.macd(close)

        result = {
            "sma_20": _last(sma20),
            "sma_50": _last(sma50),
            "sma_200": _last(sma200),
            "ema_20": _last(ema20),
            "rsi_14": _last(rsi14),
            "macd": _last(macd_line),
            "macd_signal": _last(macd_signal),
        }
        logger.info("Indicators calculated | %s", {k: v for k, v in result.items()})
        return result


def _last(series: pd.Series):
    """Return the last non-null value of a series, rounded, or None."""
    series = series.dropna()
    if series.empty:
        return None
    return round(float(series.iloc[-1]), 2)
