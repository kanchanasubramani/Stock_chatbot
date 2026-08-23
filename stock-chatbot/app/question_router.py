"""
question_router.py
-------------------
The single biggest cost lever in this app. Before ever calling the LLM,
we check whether the question is a simple factual lookup that already
exists in the stock analysis dict ("What is the RSI?", "What's the P/E?").
If so, we answer directly from Python — zero tokens spent.

Only questions that require interpretation/reasoning fall through to the
LLM (chatbot.py).
"""

from app import config

logger = config.get_logger(__name__)

# keyword -> (section, field, display label, unit)
# Order matters: more specific keywords should be checked before generic ones.
_FACT_MAP = [
    (["rsi"], "technical", "rsi", "RSI", ""),
    (["macd"], "technical", "macd", "MACD signal", ""),
    (["trend"], "technical", "trend", "Trend", ""),
    (["momentum"], "technical", "momentum", "Momentum", ""),
    (["sma 20", "sma20", "20-day", "20 day sma"], "technical", "sma_20", "SMA 20", ""),
    (["sma 50", "sma50", "50-day", "50 day sma"], "technical", "sma_50", "SMA 50", ""),
    (["sma 200", "sma200", "200-day", "200 day sma"], "technical", "sma_200", "SMA 200", ""),
    (["volatility"], "market", "volatility", "Volatility", "%"),
    (["price", "trading at", "current price"], "market", "price", "Price", "$"),
    (["p/e", "pe ratio", "price to earnings", " pe "], "fundamental", "pe", "P/E ratio", ""),
    (["peg"], "fundamental", "peg", "PEG ratio", ""),
    (["roe", "return on equity"], "fundamental", "roe", "ROE", "%"),
    (["revenue growth"], "fundamental", "revenue_growth", "Revenue growth", "%"),
    (["earnings growth"], "fundamental", "earnings_growth", "Earnings growth", "%"),
    (["sector"], "fundamental", "sector", "Sector", ""),
    (["industry"], "fundamental", "industry", "Industry", ""),
]

# Questions containing these words almost always need reasoning, even if
# they also mention a fact keyword (e.g. "why is the RSI high?").
_REASONING_SIGNALS = [
    "why", "explain", "risk", "summarize", "summarise", "should i",
    "recommend", "advice", "compare", "outlook", "important", "mean",
    "matter", "worth", "opinion", "think",
]


class QuestionRouter:
    """Decides whether a question can be answered directly from Python."""

    @staticmethod
    def answer_directly(question: str, analysis: dict) -> str | None:
        """
        Return a plain-text answer if this is a simple factual question
        answerable from `analysis`, otherwise return None (meaning: send
        to the LLM instead).
        """
        q = f" {question.lower().strip()} "

        if any(signal in q for signal in _REASONING_SIGNALS):
            return None

        for keywords, section, field, label, unit in _FACT_MAP:
            if any(kw in q for kw in keywords):
                value = analysis.get(section, {}).get(field)
                if value is None:
                    return None  # fall through to LLM, data not available
                logger.info("Question routed to Python | field=%s.%s", section, field)
                return f"{label}: {value}{unit}"

        return None
