"""
app.py
------
Streamlit entry point. Wires together StockAnalyzer, QuestionRouter,
Chatbot, ConversationManager, and guardrails behind a simple chat UI.

Run with:
    streamlit run app.py
"""
import os
from pathlib import Path      # ← add this line

from dataclasses import dataclass

from dotenv import load_dotenv

from app.logger import get_logger 

import time

import streamlit as st

from app import config, guardrails
from app.chatbot import Chatbot, ChatbotError
from app.conversation import ConversationManager, ConversationState
from app.fundamentals import FundamentalsError
from app.guardrails import GuardrailError
from app.market_data import InvalidTickerError, MarketDataError, RateLimitError
from app.question_router import QuestionRouter
from app.stock_analyzer import StockAnalyzer

logger = config.get_logger(__name__)

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


def get_conversation_manager() -> ConversationManager:
    # Not @st.cache_resource: it just wraps the already-cached Chatbot, and
    # constructing it is free, so no need to fight Streamlit's caching over
    # a mutable window_turns override.
    return ConversationManager(get_chatbot())


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None          # compact dict from StockAnalyzer
if "symbol" not in st.session_state:
    st.session_state.symbol = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # list of {"role", "content"} for DISPLAY only
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = ConversationState()  # rolling LLM memory
if "last_question_ts" not in st.session_state:
    st.session_state.last_question_ts = None  # for guardrails.check_rate_limit
if "question_count" not in st.session_state:
    st.session_state.question_count = 0       # for guardrails.check_session_budget
if "last_analyze_ts" not in st.session_state:
    st.session_state.last_analyze_ts = None   # throttles the Analyze button itself


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
symbol_input_raw = st.text_input("Stock Symbol", placeholder="e.g. TSLA, AAPL, IBM")
analyze_clicked = st.button("Analyze")

if analyze_clicked and not symbol_input_raw.strip():
    st.warning("Please enter a stock symbol first.")

elif analyze_clicked:
    try:
        # Single source of truth for ticker format — was previously
        # duplicated with a slightly different regex right in this file.
        symbol_input = guardrails.validate_ticker(symbol_input_raw)

        # Analyze hits Alpha Vantage directly (free tier: 5 req/min), so it
        # gets its own, more generous throttle independent of the chat
        # question rate limit below.
        guardrails.check_rate_limit(st.session_state.last_analyze_ts)
        st.session_state.last_analyze_ts = time.time()

        analyzer = get_stock_analyzer()
        with st.spinner(f"Analyzing {symbol_input}..."):
            result = analyzer.analyze(symbol_input)
            st.session_state.analysis = result
            st.session_state.symbol = symbol_input
            st.session_state.chat_history = []  # fresh conversation for a new ticker
            st.session_state.conversation_state = ConversationState()  # fresh LLM memory too
            st.session_state.question_count = 0

    except GuardrailError as exc:
        st.error(str(exc))
    except InvalidTickerError:
        st.error(f"'{symbol_input_raw}' doesn't look like a valid ticker. Please check and try again.")
    except RateLimitError:
        st.warning("Alpha Vantage rate limit reached. Please wait a minute and try again.")
    except (MarketDataError, FundamentalsError) as exc:
        st.error(f"Couldn't fetch data for {symbol_input_raw}: {exc}")
    except Exception:
        logger.exception("Unexpected error analyzing %s", symbol_input_raw)
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
    question_raw = st.chat_input("Ask about this stock...")
    if question_raw:
        try:
            # Length/emptiness check — replaces the old bare
            # question[:MAX_INPUT_CHARS] truncation, which silently
            # accepted anything instead of telling the user why it was cut.
            question = guardrails.validate_question(question_raw)

            # Per-session cost controls, both previously defined but unused.
            guardrails.check_rate_limit(st.session_state.last_question_ts)
            guardrails.check_session_budget(st.session_state.question_count)

            # Content moderation — fails open (logs + continues) if the
            # moderation call itself errors, per guardrails.moderate_text.
            guardrails.moderate_text(config.OPENAI_API_KEY, question)

        except GuardrailError as exc:
            st.error(str(exc))
        else:
            st.session_state.last_question_ts = time.time()
            st.session_state.question_count += 1

            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            # --- Cost optimization: try Python first, only call the LLM if needed ---
            direct_answer = QuestionRouter.answer_directly(question, analysis)

            if direct_answer:
                # Factual/Python-routed answers are raw data points, not
                # interpretation — no disclaimer needed, and skip
                # conversation memory so the rolling summary only tracks
                # turns that actually involved reasoning.
                answer = direct_answer
            else:
                chatbot = get_chatbot()
                conv_manager = get_conversation_manager()
                conversation_context = conv_manager.build_context_block(
                    st.session_state.conversation_state
                )
                try:
                    with st.spinner("Thinking..."):
                        answer = chatbot.ask(
                            symbol,
                            analysis,
                            question,
                            conversation_context=conversation_context,
                        )
                    answer = guardrails.ensure_disclaimer(answer)
                    st.session_state.conversation_state = conv_manager.add_turn(
                        st.session_state.conversation_state, question, answer
                    )
                except ChatbotError as exc:
                    answer = str(exc)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)