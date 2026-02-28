"""
embedding.py
------------
Generates dense vector embeddings using OpenAI's text-embedding-3-small model.
Includes batching, retry logic, and a simple cache.
"""
from __future__ import annotations

import os
import logging
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536  # text-embedding-3-small dimension
BATCH_SIZE = 100       # OpenAI allows up to 2048 texts per request, but 100 is safe
MAX_RETRIES = 3
RETRY_DELAY = 2.0      # seconds


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def embed_texts(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    client: Optional[OpenAI] = None,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Embed a list of texts in batches.

    Returns a list of float vectors, same length as *texts*.
    """
    if not texts:
        return []

    if client is None:
        client = get_client()

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]

        if show_progress:
            end = min(batch_start + BATCH_SIZE, len(texts))
            logger.info(f"  Embedding texts {batch_start + 1}–{end} / {len(texts)}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=model,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Embedding failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc
                logger.warning(
                    f"  Attempt {attempt} failed ({exc}). "
                    f"Retrying in {RETRY_DELAY}s…"
                )
                time.sleep(RETRY_DELAY * attempt)

    return all_embeddings


def embed_query(
    query: str,
    model: str = EMBEDDING_MODEL,
    client: Optional[OpenAI] = None,
) -> list[float]:
    """Embed a single query string."""
    results = embed_texts([query], model=model, client=client)
    return results[0]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_texts = [
        "Criminal governance in Latin America",
        "Extortion of small businesses in El Salvador",
        "Drug cartels and state protection rackets in Mexico",
    ]

    print(f"Testing embeddings with model: {EMBEDDING_MODEL}")
    embeddings = embed_texts(test_texts, show_progress=True)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    print(f"First vector (first 5 values): {embeddings[0][:5]}")
