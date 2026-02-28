"""
retrieval.py
------------
Semantic search over the ChromaDB vector store.
Returns top-k chunks with similarity scores and full metadata.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from dotenv import load_dotenv

from src.embedding import embed_query, get_client
from src.vectorstore import (
    CHROMA_PERSIST_DIR,
    COLLECTION_LARGE,
    COLLECTION_SMALL,
    get_chroma_client,
    get_or_create_collection,
)

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity search, with score and metadata."""

    chunk_id: str
    text: str
    score: float          # cosine similarity (0–1, higher = more similar)
    paper_id: str = ""
    chunk_index: int = 0
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    doi: str = ""
    topics: list[str] = field(default_factory=list)
    filename: str = ""
    token_count: int = 0

    def citation(self) -> str:
        """Short APA-style citation string."""
        authors_str = "; ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{authors_str} ({self.year}). {self.title}."

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "topics": self.topics,
            "filename": self.filename,
            "token_count": self.token_count,
        }


def _parse_retrieved(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
) -> list[RetrievedChunk]:
    """Convert raw ChromaDB results into RetrievedChunk objects."""
    results = []
    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - dist/2 → range [0, 1]
        similarity = max(0.0, 1.0 - dist / 2.0)

        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            text=doc,
            score=round(similarity, 4),
            paper_id=meta.get("paper_id", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            title=meta.get("title", ""),
            authors=json.loads(meta.get("authors", "[]")),
            year=int(meta.get("year", 0)),
            venue=meta.get("venue", ""),
            doi=meta.get("doi", ""),
            topics=json.loads(meta.get("topics", "[]")),
            filename=meta.get("filename", ""),
            token_count=int(meta.get("token_count", 0)),
        )
        results.append(chunk)
    return results


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    strategy: str = "small",
    persist_dir: str = CHROMA_PERSIST_DIR,
    openai_client=None,
    chroma_client: Optional[chromadb.ClientAPI] = None,
) -> list[RetrievedChunk]:
    """
    Perform semantic search for *query* against the vector store.

    Args:
        query: natural-language search string
        top_k: number of results to return
        strategy: "small" (256-tok chunks) or "large" (1024-tok chunks)
        persist_dir: ChromaDB persistence directory
        openai_client: reuse an existing OpenAI client (optional)
        chroma_client: reuse an existing ChromaDB client (optional)

    Returns:
        List of RetrievedChunk sorted by descending similarity.
    """
    collection_name = COLLECTION_SMALL if strategy == "small" else COLLECTION_LARGE

    if chroma_client is None:
        chroma_client = get_chroma_client(persist_dir)

    collection = get_or_create_collection(chroma_client, collection_name)

    if collection.count() == 0:
        raise RuntimeError(
            f"Collection '{collection_name}' is empty. "
            "Run `python -m src.vectorstore` first to index the papers."
        )

    if openai_client is None:
        openai_client = get_client()

    query_embedding = embed_query(query, client=openai_client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Results are wrapped in outer lists (one per query)
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = _parse_retrieved(ids, documents, metadatas, distances)
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def search_with_filter(
    query: str,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    topics_include: Optional[list[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    strategy: str = "small",
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> list[RetrievedChunk]:
    """
    Search with optional metadata filters.
    ChromaDB where-clauses applied before similarity ranking.
    """
    collection_name = COLLECTION_SMALL if strategy == "small" else COLLECTION_LARGE
    chroma_client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(chroma_client, collection_name)

    where: dict = {}
    conditions = []

    if year_min is not None:
        conditions.append({"year": {"$gte": year_min}})
    if year_max is not None:
        conditions.append({"year": {"$lte": year_max}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    openai_client = get_client()
    query_embedding = embed_query(query, client=openai_client)

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = _parse_retrieved(ids, documents, metadatas, distances)
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def deduplicate_by_paper(
    chunks: list[RetrievedChunk],
    max_per_paper: int = 2,
) -> list[RetrievedChunk]:
    """
    Keep at most *max_per_paper* chunks from the same paper.
    Preserves ranking order.
    """
    counts: dict[str, int] = {}
    kept = []
    for chunk in chunks:
        pid = chunk.paper_id
        counts[pid] = counts.get(pid, 0) + 1
        if counts[pid] <= max_per_paper:
            kept.append(chunk)
    return kept


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_query = "¿Cómo afecta la extorsión a las pequeñas empresas en América Latina?"
    print(f"Query: {test_query}\n")

    for strat in ("small", "large"):
        print(f"--- Strategy: {strat} ---")
        try:
            results = search(test_query, top_k=3, strategy=strat)
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r.score:.3f}] {r.title[:70]}")
                print(f"       {r.text[:120].strip()!r}")
        except RuntimeError as e:
            print(f"  {e}")
        print()
