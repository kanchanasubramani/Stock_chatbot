"""
analysis.py
------------
Turns raw indicator numbers into a compact, human-meaningful structured
result (trend, momentum, macd signal, volatility). This is still pure
Python/deterministic — no LLM calls happen here. The dict this module
produces is the smallest useful summary of "what's going on" with the
stock, and it's what eventually gets serialized for the LLM.
"""

import pandas as pd

from app import config

logger = config.get_logger(__name__)


class Analysis:
    """Interprets price data + indicators into a compact analysis dict."""

    @staticmethod
    def daily_returns(df: pd.DataFrame) -> pd.Series:
        return df["Close"].pct_change().dropna()

    @staticmethod
    def volatility(df: pd.DataFrame) -> float | None:
        """Annualized volatility (%) from daily returns' standard deviation."""
        returns = Analysis.daily_returns(df)
        if returns.empty:
            return None
        daily_std = returns.std()
        annualized_pct = daily_std * (252 ** 0.5) * 100
        return round(float(annualized_pct), 2)

    @staticmethod
    def trend(sma_50: float | None, sma_200: float | None) -> str:
        if sma_50 is None or sma_200 is None:
            return "unknown"
        if sma_50 > sma_200:
            return "bullish"
        if sma_50 < sma_200:
            return "bearish"
        return "neutral"

    @staticmethod
    def rsi_interpretation(rsi_value: float | None) -> str:
        if rsi_value is None:
            return "unknown"
        if rsi_value >= 70:
            return "overbought"
        if rsi_value <= 30:
            return "oversold"
        return "neutral"

    @staticmethod
    def macd_interpretation(macd_value: float | None, macd_signal: float | None) -> str:
        if macd_value is None or macd_signal is None:
            return "unknown"
        return "bullish" if macd_value > macd_signal else "bearish"

    @classmethod
    def analyze(cls, df: pd.DataFrame, indicators: dict) -> dict:
        """
        df: processed price DataFrame (used only for price/volatility here)
        indicators: output of Indicators.calculate_all()
        """
        latest_price = round(float(df["Close"].iloc[-1]), 2) if len(df) else None

        result = {
            "price": latest_price,
            "trend": cls.trend(indicators.get("sma_50"), indicators.get("sma_200")),
            "rsi": indicators.get("rsi_14"),
            "momentum": cls.rsi_interpretation(indicators.get("rsi_14")),
            "macd": cls.macd_interpretation(indicators.get("macd"), indicators.get("macd_signal")),
            "volatility": cls.volatility(df),
        }
        logger.info("Analysis complete | %s", result)
        return result
