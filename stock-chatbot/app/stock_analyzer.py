"""
stock_analyzer.py
------------------
The orchestrator. Ties MarketData -> DataProcessor -> Indicators ->
Analysis -> Fundamentals together and returns ONE compact dictionary
for a symbol. This is the only object app.py needs to talk to in order
to go from "TSLA" to a finished analysis.
"""

from app import config
from app.analysis import Analysis
from app.data_processor import DataProcessor
from app.fundamentals import Fundamentals, FundamentalsError
from app.indicators import Indicators
from app.market_data import MarketData, MarketDataError

logger = config.get_logger(__name__)


class StockAnalyzer:
    """Orchestrates the full pipeline for one ticker."""

    def __init__(self):
        self.market_data = MarketData()
        self.data_processor = DataProcessor()
        self.fundamentals = Fundamentals()

    def analyze(self, symbol: str) -> dict:
        symbol = symbol.strip().upper()

        raw_series = self.market_data.fetch_daily(symbol)
        df = self.data_processor.to_dataframe(raw_series)

        if df.empty:
            raise MarketDataError(f"No usable price data for {symbol}.")

        indicator_values = Indicators.calculate_all(df)
        market_analysis = Analysis.analyze(df, indicator_values)

        # Fundamentals are "best effort" — a failure here shouldn't block
        # showing the technical analysis, since that's the core feature.
        try:
            fundamental_data = self.fundamentals.fetch(symbol)
        except FundamentalsError as exc:
            logger.warning("Fundamentals unavailable | symbol=%s | %s", symbol, exc)
            fundamental_data = {}

        result = {
            "symbol": symbol,
            "market": {
                "price": market_analysis["price"],
                "volatility": market_analysis["volatility"],
            },
            "technical": {
                "trend": market_analysis["trend"],
                "rsi": market_analysis["rsi"],
                "momentum": market_analysis["momentum"],
                "macd": market_analysis["macd"],
                "sma_20": indicator_values.get("sma_20"),
                "sma_50": indicator_values.get("sma_50"),
                "sma_200": indicator_values.get("sma_200"),
            },
            "fundamental": {
                "pe": fundamental_data.get("pe"),
                "peg": fundamental_data.get("peg"),
                "roe": fundamental_data.get("roe"),
                "revenue_growth": fundamental_data.get("revenue_growth"),
                "earnings_growth": fundamental_data.get("earnings_growth"),
                "sector": fundamental_data.get("sector"),
                "industry": fundamental_data.get("industry"),
                "name": fundamental_data.get("name"),
            },
        }

        logger.info("Stock analysis complete | symbol=%s", symbol)
        return result
