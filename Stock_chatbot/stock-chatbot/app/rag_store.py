"""
rag_store.py
------------
Lightweight Retrieval-Augmented Generation (RAG) layer for company
documents (10-Ks, earnings-call transcripts, investor letters, etc.)
stored as PDFs on disk, one folder per ticker:

    data/documents/
        TSLA/
            10k_2024.pdf
            q2_2025_earnings_call.pdf
        AAPL/
            10k_2024.pdf

TOKEN / COST OPTIMIZATION IS THE PRIORITY HERE. Four things do the work:

1. Embed once, reuse forever.
   Every chunk's embedding is cached to disk, keyed by a hash of the chunk
   itself. Restarting the app, or asking a second question about the same
   stock, never re-pays for an embedding that's already been computed.
   `ingest()` returns the count of *newly* embedded chunks — 0 on a warm
   cache means that call cost zero embedding tokens.

2. Never send a whole document to the LLM.
   Documents are chunked (~config.RAG_CHUNK_SIZE_TOKENS tokens each) and
   only the top-k chunks most similar to the *current question* are ever
   placed in a chatbot prompt — see retrieve().

3. Hard token budget on retrieved context.
   config.RAG_MAX_CONTEXT_TOKENS caps the total size of what gets injected
   into the prompt, measured with tiktoken (real tokens, not characters),
   regardless of how many chunks scored well.

4. Cheap embedding model + cheap chunk size, both overridable via .env.

This module intentionally skips a vector-DB dependency (FAISS / Chroma /
pgvector). For a handful of PDFs per ticker, brute-force cosine similarity
over a small numpy array is fast, adds zero infra, and is easy to audit —
worth revisiting only if the per-ticker document count grows into the
thousands.

Security note: retrieved chunk text originates from files on disk, not
from the operator, so it is UNTRUSTED input from the LLM's point of view
just like PDFs uploaded by an end user. This module does not sanitize or
delimiter-wrap chunk text itself — that's guardrails.py's job
(sanitize_rag_chunks / wrap_rag_context), applied by the caller right
before the chunks go into a prompt. Keeping that responsibility in
guardrails.py keeps all prompt-injection defense in one auditable place.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import tiktoken
from openai import OpenAI, OpenAIError
from pypdf import PdfReader

from app import config

logger = config.get_logger(__name__)

# Loaded lazily (not at import time) so importing this module never requires
# network access — tiktoken fetches its BPE file on first use and caches it.
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


@dataclass
class Chunk:
    text: str
    source_file: str
    page: int
    chunk_id: str = field(default="")

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = _hash_text(f"{self.source_file}:{self.page}:{self.text}")


class RAGError(Exception):
    pass


class DocumentStore:
    """One DocumentStore per ticker: ingestion (PDF -> chunks -> cached
    embeddings) and retrieval (question -> top-k relevant chunks)."""

    def __init__(self, symbol: str, api_key: str | None = None):
        self.symbol = symbol.strip().upper()
        self.api_key = api_key or config.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)

        self.docs_dir = os.path.join(config.DOCUMENTS_DIR, self.symbol)
        self.cache_path = os.path.join(config.EMBEDDINGS_CACHE_DIR, f"{self.symbol}.json")
        os.makedirs(config.EMBEDDINGS_CACHE_DIR, exist_ok=True)

        self._cache: dict[str, dict] = self._load_cache()

    # ------------------------------------------------------------------
    # Disk cache — the main cost saver: never re-embed unchanged text
    # ------------------------------------------------------------------
    def _load_cache(self) -> dict:
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("RAG cache unreadable, rebuilding | symbol=%s | %s", self.symbol, exc)
            return {}

    def _save_cache(self) -> None:
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, separators=(",", ":"))
        os.replace(tmp_path, self.cache_path)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_pages(pdf_path: str) -> list[str]:
        reader = PdfReader(pdf_path)
        return [page.extract_text() or "" for page in reader.pages]

    @staticmethod
    def _chunk_page(text: str, source_file: str, page_num: int) -> list["Chunk"]:
        """Greedy token-based chunking with overlap, splitting on paragraph
        boundaries so chunks stay semantically coherent instead of cutting
        mid-sentence."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        max_tokens = config.RAG_CHUNK_SIZE_TOKENS
        overlap_tokens = config.RAG_CHUNK_OVERLAP_TOKENS

        for para in paragraphs:
            para_tokens = _count_tokens(para)

            if current_tokens + para_tokens > max_tokens and current:
                chunks.append(Chunk("\n".join(current), source_file, page_num))
                # carry a small overlap tail forward for continuity
                overlap_lines: list[str] = []
                overlap_count = 0
                for line in reversed(current):
                    t = _count_tokens(line)
                    if overlap_count + t > overlap_tokens:
                        break
                    overlap_lines.insert(0, line)
                    overlap_count += t
                current, current_tokens = overlap_lines, overlap_count

            current.append(para)
            current_tokens += para_tokens

        if current:
            chunks.append(Chunk("\n".join(current), source_file, page_num))
        return chunks

    def ingest(self, force: bool = False) -> int:
        """Chunk every PDF in this ticker's folder and make sure each chunk
        has an embedding (fresh or cached). Returns the number of *newly*
        embedded chunks — 0 means this call cost zero embedding tokens."""
        if not os.path.isdir(self.docs_dir):
            logger.info("No documents folder for symbol | symbol=%s", self.symbol)
            return 0

        new_embeddings = 0
        seen_ids: set[str] = set()

        for filename in sorted(os.listdir(self.docs_dir)):
            if not filename.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(self.docs_dir, filename)

            try:
                pages = self._extract_pages(pdf_path)
            except Exception as exc:  # malformed/encrypted PDF shouldn't kill ingestion
                logger.error("Failed to read PDF | symbol=%s | file=%s | %s",
                             self.symbol, filename, exc)
                continue

            for page_num, page_text in enumerate(pages, start=1):
                if not page_text.strip():
                    continue
                for chunk in self._chunk_page(page_text, filename, page_num):
                    seen_ids.add(chunk.chunk_id)
                    if not force and chunk.chunk_id in self._cache:
                        continue  # already embedded — zero cost
                    embedding = self._embed(chunk.text)
                    self._cache[chunk.chunk_id] = {
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "page": chunk.page,
                        "embedding": embedding,
                        "ingested_at": time.time(),
                    }
                    new_embeddings += 1

        # Drop cache entries for chunks that no longer exist (deleted/edited PDFs)
        stale = [cid for cid in self._cache if cid not in seen_ids]
        for cid in stale:
            del self._cache[cid]

        if new_embeddings or stale:
            self._save_cache()

        logger.info(
            "RAG ingestion complete | symbol=%s | new_embeddings=%d | stale_removed=%d | total_chunks=%d",
            self.symbol, new_embeddings, len(stale), len(self._cache),
        )
        return new_embeddings

    def _embed(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model=config.OPENAI_EMBEDDING_MODEL,
                input=text,
            )
        except OpenAIError as exc:
            logger.error("Embedding request failed | symbol=%s | %s", self.symbol, type(exc).__name__)
            raise RAGError("Could not generate embeddings for document ingestion.") from exc
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k most relevant chunks for `question`, trimmed to
        fit config.RAG_MAX_CONTEXT_TOKENS in total. Returns [] if this
        symbol has no ingested documents — callers should treat that as
        "no RAG context available" and fall back to the existing
        analysis-only prompt."""
        if not self._cache:
            return []

        top_k = top_k or config.RAG_TOP_K
        query_embedding = np.array(self._embed(question))

        scored = []
        for chunk_id, entry in self._cache.items():
            chunk_embedding = np.array(entry["embedding"])
            score = _cosine_similarity(query_embedding, chunk_embedding)
            if score >= config.RAG_MIN_SIMILARITY:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict] = []
        token_budget = config.RAG_MAX_CONTEXT_TOKENS

        for score, entry in scored[: top_k * 2]:  # small candidate pool, trimmed by token budget below
            tokens = _count_tokens(entry["text"])
            if tokens > token_budget:
                if not results:
                    # Guarantee at least something useful on the very first
                    # hit, truncated to fit, rather than returning nothing.
                    encoding = _get_encoding()
                    truncated_ids = encoding.encode(entry["text"])[:token_budget]
                    results.append({
                        "text": encoding.decode(truncated_ids),
                        "source_file": entry["source_file"],
                        "page": entry["page"],
                        "score": round(float(score), 3),
                    })
                break
            results.append({
                "text": entry["text"],
                "source_file": entry["source_file"],
                "page": entry["page"],
                "score": round(float(score), 3),
            })
            token_budget -= tokens
            if len(results) >= top_k:
                break

        logger.info(
            "RAG retrieval | symbol=%s | question_len=%d | chunks_returned=%d | tokens_used=%d",
            self.symbol, len(question), len(results),
            config.RAG_MAX_CONTEXT_TOKENS - token_budget,
        )
        return results


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Convenience entry point used by chatbot.py / stock_analyzer.py
# ---------------------------------------------------------------------------
_store_cache: dict[str, DocumentStore] = {}


def get_relevant_context(symbol: str, question: str) -> list[dict]:
    """Lazily create + ingest a DocumentStore for `symbol` the first time
    it's needed (ingest() is a no-op / zero-cost once the on-disk embedding
    cache is warm), then retrieve the chunks relevant to `question`.

    Only call this for questions that are already going to the LLM (i.e.
    after question_router.answer_directly() returns None) — factual
    lookups should keep costing zero tokens, RAG or not."""
    symbol = symbol.strip().upper()
    store = _store_cache.get(symbol)
    if store is None:
        store = DocumentStore(symbol)
        store.ingest()
        _store_cache[symbol] = store
    return store.retrieve(question)
