"""
vectorstore.py
--------------
ChromaDB vector store: initialize, populate, persist, and query.
Each collection corresponds to one chunking strategy (small / large).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from src.chunking import Chunk, chunk_papers
from src.embedding import embed_texts, get_client
from src.ingestion import load_papers

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent.parent / "chroma_db"),
)

COLLECTION_SMALL = "papers_small"   # 256-token chunks
COLLECTION_LARGE = "papers_large"   # 1024-token chunks


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def get_chroma_client(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.ClientAPI,
    name: str,
) -> chromadb.Collection:
    """Get or create a named collection (cosine similarity)."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _metadata_for_chroma(chunk: Chunk) -> dict:
    """Convert Chunk to flat metadata dict (ChromaDB requires flat, scalar values)."""
    return {
        "paper_id": chunk.paper_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "authors": json.dumps(chunk.authors),
        "year": chunk.year,
        "venue": chunk.venue,
        "doi": chunk.doi,
        "topics": json.dumps(chunk.topics),
        "filename": chunk.filename,
        "token_count": chunk.token_count,
    }


def index_chunks(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    batch_size: int = 50,
    show_progress: bool = True,
    openai_client=None,
) -> None:
    """Embed and upsert *chunks* into a ChromaDB collection."""
    if openai_client is None:
        openai_client = get_client()
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        end = min(start + batch_size, total)

        if show_progress:
            logger.info(f"  Indexing chunks {start + 1}–{end} / {total}")

        texts = [c.text for c in batch]
        embeddings = embed_texts(texts, client=openai_client, show_progress=False)

        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[_metadata_for_chroma(c) for c in batch],
        )

    if show_progress:
        logger.info(f"  Indexed {total} chunks into collection '{collection.name}'.")


def build_index(
    strategy: str = "small",
    persist_dir: str = CHROMA_PERSIST_DIR,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """
    Build (or reuse) a ChromaDB collection for the given chunking strategy.

    Args:
        strategy: "small" (256 tok) or "large" (1024 tok)
        persist_dir: path for ChromaDB persistence
        force_rebuild: if True, delete the collection and re-index
    """
    collection_name = COLLECTION_SMALL if strategy == "small" else COLLECTION_LARGE

    chroma = get_chroma_client(persist_dir)

    if force_rebuild:
        try:
            chroma.delete_collection(collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'.")
        except Exception:
            pass

    collection = get_or_create_collection(chroma, collection_name)

    if collection.count() > 0 and not force_rebuild:
        logger.info(
            f"Collection '{collection_name}' already has {collection.count()} chunks. "
            "Skipping re-index. Pass force_rebuild=True to re-index."
        )
        return collection

    logger.info("Loading papers…")
    papers = load_papers(verbose=True)

    logger.info(f"Chunking with strategy='{strategy}'…")
    chunks = chunk_papers(papers, strategy=strategy)
    logger.info(f"  {len(chunks)} chunks to index.")

    index_chunks(chunks, collection, show_progress=True)
    return collection


def build_all_indexes(
    persist_dir: str = CHROMA_PERSIST_DIR,
    force_rebuild: bool = False,
) -> dict[str, chromadb.Collection]:
    """Build both small and large index collections."""
    return {
        "small": build_index("small", persist_dir, force_rebuild),
        "large": build_index("large", persist_dir, force_rebuild),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Build ChromaDB index")
    parser.add_argument(
        "--strategy",
        choices=["small", "large", "all"],
        default="all",
        help="Chunking strategy to index",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and rebuild the collection",
    )
    args = parser.parse_args()

    if args.strategy == "all":
        cols = build_all_indexes(force_rebuild=args.force_rebuild)
        for name, col in cols.items():
            print(f"Collection '{name}': {col.count()} chunks")
    else:
        col = build_index(args.strategy, force_rebuild=args.force_rebuild)
        print(f"Collection '{args.strategy}': {col.count()} chunks")
