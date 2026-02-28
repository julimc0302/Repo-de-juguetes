"""
rag.py — RAG query engine: embed question → retrieve chunks → prompt → GPT-4o answer.
"""

import os
import json
from pathlib import Path

import chromadb
import openai
from dotenv import load_dotenv

load_dotenv()

try:
    from config import (
        OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL,
        CHROMA_PATH, TOP_K,
    )
    from prompts import build_messages, STRATEGIES
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHROMA_PATH = "./chroma_db"
    TOP_K = 5
    from prompts import build_messages, STRATEGIES

openai.api_key = OPENAI_API_KEY
_openai_client = None


def _get_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ── ChromaDB ──────────────────────────────────────────────────────────────────

_chroma_client = None
_collections: dict[int, chromadb.Collection] = {}


def load_collections() -> dict[int, chromadb.Collection]:
    """Open ChromaDB and return dict {chunk_size: collection}."""
    global _chroma_client, _collections
    if _collections and any(v is not None for v in _collections.values()):
        return _collections

    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    for size in [256, 1024]:
        try:
            col = _chroma_client.get_collection(f"papers_{size}")
            _collections[size] = col
        except Exception:
            # Collection doesn't exist yet (not yet ingested)
            _collections[size] = None
    return _collections


def clear_collections_cache() -> None:
    """Reset ChromaDB client and collection cache (call after re-ingestion)."""
    global _chroma_client, _collections
    _chroma_client = None
    _collections.clear()


def query_chromadb(
    collection: chromadb.Collection,
    question_embedding: list[float],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Query a ChromaDB collection with an embedding vector.

    Returns a list of dicts: {text, metadata, distance}.
    """
    if collection is None:
        return []

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append(
            {
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return chunks


# ── Formatting ────────────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks for display (used by prompts internally via _build_context_text)."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        parts.append(
            f"[Chunk {i}] {meta.get('title', 'Unknown')} "
            f"(by {meta.get('authors', 'Unknown')}, {meta.get('year', '')})\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(parts)


def generate_apa_citation(metadata: dict) -> str:
    """Generate an APA-style citation from chunk metadata."""
    authors = metadata.get("authors", "Unknown Author")
    year = metadata.get("year") or "n.d."
    title = metadata.get("title", "Untitled")
    filename = metadata.get("filename", "")
    return f"{authors} ({year}). {title}. Retrieved from {filename}"


# ── Main ask function ─────────────────────────────────────────────────────────

def ask(
    question: str,
    chunk_size: int = 256,
    strategy: int = 1,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question.
    2. Retrieve top_k chunks from ChromaDB.
    3. Build prompt messages using the chosen strategy.
    4. Call GPT-4o.
    5. Return answer, APA citations, and retrieved chunks.

    Returns:
        {
            "answer": str,
            "citations": list[str],
            "chunks": list[dict],
            "strategy": int,
            "chunk_size": int,
        }
    """
    if chat_history is None:
        chat_history = []

    client = _get_openai_client()

    # 1. Embed question
    embed_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )
    question_embedding = embed_response.data[0].embedding

    # 2. Retrieve chunks
    collections = load_collections()
    collection = collections.get(chunk_size)
    chunks = query_chromadb(collection, question_embedding, top_k=TOP_K)

    if not chunks:
        return {
            "answer": (
                "No indexed content found. Please run ingestion first: `python ingest.py`."
            ),
            "citations": [],
            "chunks": [],
            "strategy": strategy,
            "chunk_size": chunk_size,
        }

    # 3. Build messages
    messages = build_messages(question, chunks, chat_history, strategy=strategy)

    # 4. Call GPT-4o
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
    )
    answer_text = response.choices[0].message.content.strip()

    # 5. Build APA citations (deduplicated by filename)
    seen = set()
    citations = []
    for chunk in chunks:
        fname = chunk["metadata"].get("filename", "")
        if fname not in seen:
            seen.add(fname)
            citations.append(generate_apa_citation(chunk["metadata"]))

    return {
        "answer": answer_text,
        "citations": citations,
        "chunks": chunks,
        "strategy": strategy,
        "chunk_size": chunk_size,
    }
