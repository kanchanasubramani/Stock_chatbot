"""
conversation.py
----------------
Rolling-window conversation memory for the chatbot.

Problem: resending the full chat history on every question makes token cost
grow (and compound) with conversation length. Just dropping old turns
(guardrails.trim_history does this) is cheap but loses context a user might
reasonably expect the bot to remember — "what did you say about the RSI a
few questions ago?".

Solution: keep the last `window_turns` Q&A pairs verbatim, and fold
everything older into a single running summary. The summary is
re-generated only when the window overflows — merging the *previous*
summary with the turns that just fell out of the window — via one small,
cheap LLM call (capped at ~120 output tokens). The prompt then carries
roughly constant history cost instead of cost that grows with conversation
length.

This module is stateless with respect to storage: the caller (e.g. a
Streamlit session, one per user) owns a ConversationState and persists it
between requests. Typically one ConversationState per (session, symbol).
"""

from dataclasses import dataclass, field

from openai import OpenAIError

from app import config

logger = config.get_logger(__name__)

# Small and cheap on purpose: this call only needs to preserve the gist of
# what fell out of the window, not reproduce it.
_SUMMARY_MAX_TOKENS = 120

_SUMMARIZER_SYSTEM_PROMPT = (
    "Summarize the following stock-analysis conversation turns for use as "
    "background context in later turns of the same conversation.\n"
    "- Keep only facts, figures, and conclusions that were actually stated.\n"
    "- Never invent or infer anything that wasn't said.\n"
    "- Merge with the existing summary given below rather than appending to "
    "it — produce one updated summary, not two.\n"
    "- Maximum 60 words. Plain text. No preamble, no bullet points."
)


@dataclass
class ConversationState:
    """Per-session, per-symbol conversation memory. Plain data — persist it
    however your app already manages session state."""
    summary: str = ""
    recent_turns: list[dict] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": str}]


class ConversationManager:
    """Keeps `window_turns` most recent Q&A pairs verbatim; older turns are
    folded into a rolling summary using the same Chatbot's client, so no
    extra API key/config wiring is needed."""

    def __init__(self, chatbot, window_turns: int | None = None):
        self.chatbot = chatbot
        self.window_turns = window_turns or config.CHAT_ROLLING_WINDOW_TURNS

    def add_turn(self, state: ConversationState, question: str, answer: str) -> ConversationState:
        """Call this after every chatbot.ask() to update memory. Returns the
        same state object, mutated (and re-summarized if the window just
        overflowed) — persist it back to session storage."""
        state.recent_turns.append({"role": "user", "content": question})
        state.recent_turns.append({"role": "assistant", "content": answer})

        max_messages = self.window_turns * 2  # each turn = 1 user + 1 assistant message
        if len(state.recent_turns) > max_messages:
            overflow = state.recent_turns[:-max_messages]
            state.recent_turns = state.recent_turns[-max_messages:]
            state.summary = self._summarize(overflow, state.summary)

        return state

    def _summarize(self, overflow_turns: list[dict], existing_summary: str) -> str:
        overflow_text = "\n".join(f"{t['role']}: {t['content']}" for t in overflow_turns)
        user_content = (
            f"Existing summary: {existing_summary or '(none yet)'}\n\n"
            f"New turns to fold in:\n{overflow_text}"
        )
        try:
            response = self.chatbot.client.chat.completions.create(
                model=self.chatbot.model,
                max_tokens=_SUMMARY_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": _SUMMARIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            summary = response.choices[0].message.content.strip()
            logger.info(
                "Conversation summarized | folded_turns=%d | summary_len=%d",
                len(overflow_turns), len(summary),
            )
            return summary
        except OpenAIError as exc:
            # Fail open: keep the previous summary rather than losing all
            # memory just because one summarization call errored.
            logger.warning("Conversation summarization failed, keeping previous summary | %s", exc)
            return existing_summary

    @staticmethod
    def build_context_block(state: ConversationState) -> str:
        """Render (summary + recent turns) as plain text, ready to be
        sanitized/wrapped by guardrails.wrap_conversation_context() before
        it goes into a chatbot prompt. Returns "" when there's no history
        yet, so callers can skip adding an (empty) History section."""
        if not state.summary and not state.recent_turns:
            return ""
        parts = []
        if state.summary:
            parts.append(f"Earlier in this conversation: {state.summary}")
        for t in state.recent_turns:
            role = "User" if t["role"] == "user" else "Assistant"
            parts.append(f"{role}: {t['content']}")
        return "\n".join(parts)
