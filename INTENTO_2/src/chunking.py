"""
chunking.py
-----------
Splits paper text into overlapping chunks of a given token size.
Supports two strategies: small (256 tokens) and large (1024 tokens).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import tiktoken

ENCODING_NAME = "cl100k_base"  # used by text-embedding-3-small and GPT-4


@dataclass
class Chunk:
    """A single text chunk with its provenance metadata."""

    paper_id: str
    chunk_index: int
    text: str
    token_count: int
    # Paper-level metadata mirrored for convenience
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    doi: str = ""
    topics: list[str] = field(default_factory=list)
    filename: str = ""

    @property
    def chunk_id(self) -> str:
        return f"{self.paper_id}_chunk_{self.chunk_index:04d}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_count": self.token_count,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "topics": self.topics,
            "filename": self.filename,
        }


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def chunk_text(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[tuple[str, int]]:
    """
    Split *text* into overlapping token windows.

    Returns a list of (chunk_text, token_count) tuples.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)

    if not tokens:
        return []

    step = max(1, max_tokens - overlap_tokens)
    chunks: list[tuple[str, int]] = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_decoded = enc.decode(chunk_tokens)
        chunks.append((chunk_text_decoded, len(chunk_tokens)))
        if end == len(tokens):
            break
        start += step

    return chunks


def chunk_paper(
    paper: dict,
    strategy: Literal["small", "large"] = "small",
) -> list[Chunk]:
    """
    Chunk a single paper dict (from ingestion.load_papers) into Chunk objects.

    strategy="small"  → max_tokens=256,  overlap=32
    strategy="large"  → max_tokens=1024, overlap=128
    """
    if strategy == "small":
        max_tokens, overlap = 256, 32
    else:
        max_tokens, overlap = 1024, 128

    raw_chunks = chunk_text(paper["text"], max_tokens=max_tokens, overlap_tokens=overlap)

    chunks = []
    for idx, (text, token_count) in enumerate(raw_chunks):
        c = Chunk(
            paper_id=paper["id"],
            chunk_index=idx,
            text=text,
            token_count=token_count,
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            year=paper.get("year", 0) or 0,
            venue=paper.get("venue", "") or "",
            doi=paper.get("doi", "") or "",
            topics=paper.get("topics", []),
            filename=paper.get("filename", ""),
        )
        chunks.append(c)
    return chunks


def chunk_papers(
    papers: list[dict],
    strategy: Literal["small", "large"] = "small",
) -> list[Chunk]:
    """Chunk all papers and return a flat list of Chunk objects."""
    all_chunks: list[Chunk] = []
    for paper in papers:
        paper_chunks = chunk_paper(paper, strategy=strategy)
        all_chunks.extend(paper_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.ingestion import load_papers

    papers = load_papers(verbose=False)

    for strategy in ("small", "large"):
        chunks = chunk_papers(papers, strategy=strategy)
        total_tokens = sum(c.token_count for c in chunks)
        print(
            f"Strategy '{strategy}': {len(chunks)} chunks, "
            f"{total_tokens:,} total tokens"
        )
        # Sample
        sample = chunks[0]
        print(f"  Sample chunk_id: {sample.chunk_id}")
        print(f"  First 200 chars: {sample.text[:200]!r}")
        print()
