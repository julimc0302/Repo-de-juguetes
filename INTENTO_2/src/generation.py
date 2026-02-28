"""
generation.py
-------------
Orchestrates the four prompt strategies for answer generation,
given retrieved context chunks and a user question.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval import RetrievedChunk, search
from prompts.strategy1_delimiters import build_prompt as prompt_delimiters
from prompts.strategy2_json import build_prompt as prompt_json
from prompts.strategy3_fewshot import build_prompt as prompt_fewshot
from prompts.strategy4_cot import build_prompt as prompt_cot

load_dotenv()
logger = logging.getLogger(__name__)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
StrategyName = Literal["delimiters", "json", "fewshot", "cot"]

STRATEGY_BUILDERS = {
    "delimiters": prompt_delimiters,
    "json": prompt_json,
    "fewshot": prompt_fewshot,
    "cot": prompt_cot,
}

STRATEGY_LABELS = {
    "delimiters": "Strategy 1: Delimiters & Structured Sections",
    "json": "Strategy 2: JSON Output Format",
    "fewshot": "Strategy 3: Few-Shot Examples",
    "cot": "Strategy 4: Chain-of-Thought",
}


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Convert retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        authors_str = "; ".join(chunk.authors[:3])
        if len(chunk.authors) > 3:
            authors_str += " et al."
        header = f"[{i}] {chunk.title} — {authors_str} ({chunk.year})"
        parts.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    strategy: StrategyName = "delimiters",
    model: str = CHAT_MODEL,
    client: Optional[OpenAI] = None,
    temperature: float = 0.3,
    max_tokens: int = 1200,
) -> dict:
    """
    Generate an answer for *question* using retrieved *chunks* and the
    specified prompt *strategy*.

    Returns a dict with:
        answer       : str   — the model's response
        strategy     : str   — strategy name used
        model        : str   — model ID
        usage        : dict  — token usage
        chunks_used  : int   — number of context chunks
        prompt_tokens: int
        completion_tokens: int
    """
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = _format_context(chunks)
    builder = STRATEGY_BUILDERS[strategy]
    messages = builder(question=question, context=context)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start_time

    answer = response.choices[0].message.content.strip()
    usage = response.usage

    return {
        "answer": answer,
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "model": model,
        "chunks_used": len(chunks),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "elapsed_seconds": round(elapsed, 2),
    }


def compare_strategies(
    question: str,
    chunks: list[RetrievedChunk],
    model: str = CHAT_MODEL,
    client: Optional[OpenAI] = None,
    temperature: float = 0.3,
    max_tokens: int = 1200,
) -> dict[str, dict]:
    """Run all four strategies and return results keyed by strategy name."""
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = {}
    for strategy in STRATEGY_BUILDERS:
        logger.info(f"Running strategy: {strategy}")
        results[strategy] = generate_answer(
            question=question,
            chunks=chunks,
            strategy=strategy,
            model=model,
            client=client,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return results


def rag_query(
    question: str,
    strategy: StrategyName = "delimiters",
    top_k: int = 5,
    chunk_strategy: str = "small",
    model: str = CHAT_MODEL,
    temperature: float = 0.3,
) -> dict:
    """
    Full RAG pipeline: retrieve + generate.

    Convenience function that wraps retrieval + generation in one call.
    Returns the generation result dict plus a 'retrieved_chunks' key.
    """
    chunks = search(question, top_k=top_k, strategy=chunk_strategy)
    result = generate_answer(
        question=question,
        chunks=chunks,
        strategy=strategy,
        model=model,
        temperature=temperature,
    )
    result["retrieved_chunks"] = [c.to_dict() for c in chunks]
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    question = "¿Cuáles son las principales estrategias de resistencia a la extorsión en América Latina?"

    print(f"Question: {question}\n{'=' * 70}")

    result = rag_query(question, strategy="delimiters", top_k=5)
    print(f"\n[{result['strategy_label']}]\n")
    print(result["answer"])
    print(f"\nTokens used: {result['total_tokens']} | Time: {result['elapsed_seconds']}s")
