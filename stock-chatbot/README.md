# Stock Analysis Chatbot

A Streamlit app that analyzes a stock ticker and lets you ask follow-up
questions in a chat interface — with most obvious cost cut by routing
simple factual questions straight to Python instead of the LLM.

## Setup

```bash
pip install -r requirements.txt
```

Edit `.env` and add your real keys:

```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

Get a free Alpha Vantage key at https://www.alphavantage.co/support/#api-key
(free tier: 25 requests/day, ~5 requests/minute).

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## How the files connect

```
app.py                     Streamlit UI — the only file that renders anything.
  ├── app/config.py        Env vars and constants. Everything imports this.
  ├── app/logger.py         Logging setup (format, level). config.py re-exports
  │                         get_logger() from here so nothing else had to change.
  ├── app/stock_analyzer.py  Orchestrator, called once per "Analyze" click.
  │     ├── app/market_data.py     -> Alpha Vantage TIME_SERIES_DAILY (raw JSON)
  │     ├── app/data_processor.py  -> raw JSON -> clean DataFrame
  │     ├── app/indicators.py      -> DataFrame -> SMA/EMA/RSI/MACD numbers
  │     ├── app/analysis.py        -> numbers -> trend/momentum/macd labels
  │     └── app/fundamentals.py    -> Alpha Vantage OVERVIEW -> P/E, ROE, etc.
  │
  ├── app/question_router.py   Checks each user question FIRST.
  │                             If it's a simple lookup ("what's the RSI?"),
  │                             answers directly from the analysis dict —
  │                             zero LLM calls.
  │
  └── app/chatbot.py          Only called when question_router returns None
                                (the question needs reasoning). Sends compact
                                JSON + the question to OpenAI, capped at
                                ~200 output tokens / 120 words.
```

### Request flow for "Analyze"

```
ticker string
  -> MarketData.fetch_daily()        raw Alpha Vantage JSON
  -> DataProcessor.to_dataframe()    clean pandas DataFrame
  -> Indicators.calculate_all()      {sma_20, sma_50, sma_200, rsi_14, macd, ...}
  -> Analysis.analyze()              {trend, rsi, momentum, macd, volatility}
  -> Fundamentals.fetch()            {pe, roe, revenue_growth, ...}
  -> StockAnalyzer assembles all of the above into ONE compact dict,
     stored in st.session_state.analysis
```

That final dict — never the raw DataFrame or price history — is the only
thing ever sent to the LLM, and only when a question actually needs it.

### Request flow for a follow-up question

```
user types a question
  -> QuestionRouter.answer_directly(question, analysis)
       -> match found?  answer from Python, done. (no API cost)
       -> no match?     fall through to Chatbot.ask()
                           -> compact JSON (no whitespace) + question
                           -> OpenAI chat.completions, max_tokens capped
                           -> concise answer (<=120 words)
```

## Design notes / best practices baked in

- **Separation of concerns**: each module does exactly one job (fetch,
  transform, calculate, interpret, orchestrate, route, chat, render).
  This makes it easy to test each piece in isolation and to swap out,
  e.g., Alpha Vantage for another data provider later.
- **Deterministic math stays in Python**: SMA/EMA/RSI/MACD/volatility are
  all calculated with pandas, not asked of the LLM. This is both cheaper
  and more trustworthy — the LLM can't "make up" a number.
- **Compact payloads only**: the LLM only ever sees the final ~15-field
  analysis dict, serialized without whitespace
  (`json.dumps(data, separators=(",", ":"))`). Raw OHLCV history and
  DataFrames never leave `data_processor.py` / `indicators.py`.
- **No conversation history resent**: each LLM call sends only the
  symbol, the compact analysis, and the current question — not the
  entire chat log — so cost doesn't grow with conversation length.
  (The visible chat history in the UI is for the user's benefit only.)
- **Question routing for cost control**: `question_router.py` uses
  keyword matching to catch simple factual questions ("what's the RSI",
  "what's the P/E") and answer them directly from the already-computed
  analysis dict, at zero API cost. Anything containing reasoning signals
  ("why", "explain", "risk", "summarize", etc.) is routed to the LLM.
- **Custom exceptions per module** (`InvalidTickerError`, `RateLimitError`,
  `MarketDataError`, `FundamentalsError`, `ChatbotError`) so `app.py` can
  show specific, user-friendly error messages instead of a generic crash.
- **Graceful degradation**: if fundamentals fail to load, the app still
  shows the technical analysis rather than failing the whole request.
- **Structured logging**: every module logs through `config.get_logger()`
  with a consistent format, and API keys are never included in any log
  line or LLM payload.
- **Caching**: `st.cache_resource` is used for `Chatbot` and
  `StockAnalyzer` so the OpenAI client and Alpha Vantage clients aren't
  re-instantiated on every Streamlit rerun (which happens on every
  widget interaction).
- **Fixed max output tokens + word limit** are enforced both in the
  system prompt (120 words) and via the API's `max_tokens` parameter
  (~200 tokens), so a runaway response can't happen even if the model
  ignores the prompt instruction. `chatbot.py` also post-processes the
  response and hard-truncates it to ~130 words as a code-level safety
  net — never trust the model to police its own output length.

## Guardrails

- **Prompt injection defense**: the system prompt explicitly tells the
  model that the `Question` field is untrusted user input, not
  instructions, and the user's question is wrapped in `<<<...>>>`
  delimiters so it's structurally distinguishable from real instructions.
  A cheap keyword pre-filter in `chatbot.py`
  (`_looks_like_injection`) catches obvious attempts ("ignore previous
  instructions", "you are now...", "reveal your system prompt") and
  short-circuits with a canned refusal *before* spending any tokens.
  This is a heuristic, not a guarantee — for high-stakes deployments
  you'd want a dedicated moderation/classifier step too.
- **Ticker input validation**: `app.py` validates the symbol against a
  strict regex (`^[A-Z]{1,5}(\.[A-Z]{1,2})?$`) before it's ever sent to
  Alpha Vantage — blocks garbage input, wastes zero API quota on
  obviously invalid tickers.
- **Scope restriction**: the system prompt confines the model to the
  supplied ticker/data only, so it won't wander into general financial
  advice, unrelated topics, or other stocks it wasn't given data for.
- **No personalized financial advice / no return guarantees**: baked
  into the system prompt rules directly, since this is a compliance-
  relevant boundary for anything finance-adjacent.

## Known limitations (by design, per spec)

- No comparison across multiple tickers, no news, no ML, no database,
  no FastAPI backend — single-file Streamlit app on purpose.
- Alpha Vantage's free tier is rate-limited (5 calls/min, 25/day) — the
  app surfaces this as a friendly warning rather than crashing.
- `outputsize=compact` is used (last ~100 trading days), which is enough
  for SMA 200 to have *some* data but will show `null`/`None` until 200
  daily bars exist if you switch to a freshly-listed ticker. Switch to
  `outputsize=full` in `market_data.py` if you need guaranteed SMA 200
  coverage (uses more of your API quota per call).
