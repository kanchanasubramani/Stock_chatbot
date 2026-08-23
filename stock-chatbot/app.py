"""
app.py
------
Streamlit entry point. Wires together StockAnalyzer, QuestionRouter,
and Chatbot behind a simple chat UI.

Run with:
    streamlit run app.py
"""

import re

import streamlit as st

from app import config
from app.chatbot import Chatbot, ChatbotError
from app.fundamentals import FundamentalsError
from app.market_data import InvalidTickerError, MarketDataError, RateLimitError
from app.question_router import QuestionRouter
from app.stock_analyzer import StockAnalyzer

logger = config.get_logger(__name__)

# Real ticker symbols are 1-5 letters, optionally with a dot-suffix class
# (e.g. BRK.B). Validating this before it ever reaches Alpha Vantage blocks
# garbage/injection-y input and saves a wasted API call on obvious junk.
_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$") 
# _TICKET_PATTERN can be used only in this file, not exported to other modules, 
# so it's private to this module.

st.set_page_config(page_title="Stock Analysis Chatbot", page_icon="📈")
st.title("Stock Analysis Chatbot")


# ---------------------------------------------------------------------------
# Cached resources — created once per session/process, not once per rerun.
# ---------------------------------------------------------------------------
@st.cache_resource 
# caches the return value of the function for the duration of the Streamlit session
def get_chatbot() -> Chatbot:
    return Chatbot()


@st.cache_resource
def get_stock_analyzer() -> StockAnalyzer:
    return StockAnalyzer()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None          # compact dict from StockAnalyzer
if "symbol" not in st.session_state:
    st.session_state.symbol = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # list of {"role", "content"} for DISPLAY only


# ---------------------------------------------------------------------------
# Startup config check
# ---------------------------------------------------------------------------
missing = config.validate_config()
if missing:
    #st.error() displays an error message in the Streamlit app, and 
    # st.stop() halts execution of the script.
    st.error( 
        "Missing required configuration: "
        + ", ".join(missing)
        + ". Add these to your .env file."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Step 1: Ticker input + Analyze
# ---------------------------------------------------------------------------
symbol_input = st.text_input("Stock Symbol", placeholder="e.g. TSLA, AAPL, IBM").strip().upper()
analyze_clicked = st.button("Analyze")

if analyze_clicked and symbol_input and not _TICKER_PATTERN.match(symbol_input):
    st.error("Please enter a valid ticker symbol (1-5 letters, e.g. TSLA, AAPL, BRK.B).")

elif analyze_clicked and symbol_input:
    analyzer = get_stock_analyzer()
    with st.spinner(f"Analyzing {symbol_input}..."):
        try:
            result = analyzer.analyze(symbol_input)
            st.session_state.analysis = result
            st.session_state.symbol = symbol_input
            st.session_state.chat_history = []  # fresh conversation for a new ticker
        except InvalidTickerError:
            st.error(f"'{symbol_input}' doesn't look like a valid ticker. Please check and try again.")
        except RateLimitError:
            st.warning("Alpha Vantage rate limit reached. Please wait a minute and try again.")
        except (MarketDataError, FundamentalsError) as exc:
            st.error(f"Couldn't fetch data for {symbol_input}: {exc}")
        except Exception:
            logger.exception("Unexpected error analyzing %s", symbol_input)
            st.error("Something went wrong analyzing this stock. Please try again.")


# ---------------------------------------------------------------------------
# Step 2: Show analysis summary + chat interface
# ---------------------------------------------------------------------------
if st.session_state.analysis:
    analysis = st.session_state.analysis
    symbol = st.session_state.symbol

    st.subheader(f"{symbol} — Quick Snapshot")
    m, t, f = analysis["market"], analysis["technical"], analysis["fundamental"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${m['price']}" if m["price"] is not None else "N/A")
    col2.metric("Trend", t["trend"].capitalize())
    col3.metric("RSI (14)", t["rsi"] if t["rsi"] is not None else "N/A")

    col4, col5, col6 = st.columns(3)
    col4.metric("MACD", t["macd"].capitalize())
    col5.metric("Volatility", f"{m['volatility']}%" if m["volatility"] is not None else "N/A")
    col6.metric("P/E", f["pe"] if f["pe"] is not None else "N/A")

    # One-time concise LLM summary, generated only the first time this
    # ticker is analyzed (not on every rerun) — cached in chat_history.
    if not st.session_state.chat_history:
        chatbot = get_chatbot()
        with st.spinner("Generating summary..."):
            try:
                summary = chatbot.ask(symbol, analysis, "Summarize this stock briefly.")
                st.session_state.chat_history.append({"role": "assistant", "content": summary})
            except ChatbotError as exc:
                st.session_state.chat_history.append({"role": "assistant", "content": str(exc)})

    st.divider()

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input for follow-up questions
    question = st.chat_input("Ask about this stock...")
    if question:
        question = question[: config.MAX_INPUT_CHARS]
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # --- Cost optimization: try Python first, only call the LLM if needed ---
        direct_answer = QuestionRouter.answer_directly(question, analysis)

        if direct_answer:
            answer = direct_answer
        else:
            chatbot = get_chatbot()
            try:
                with st.spinner("Thinking..."):
                    answer = chatbot.ask(symbol, analysis, question)
            except ChatbotError as exc:
                answer = str(exc)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

elif analyze_clicked and not symbol_input:
    st.warning("Please enter a stock symbol first.")
