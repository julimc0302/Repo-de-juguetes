"""
evaluation_script.py
--------------------
Evaluates the RAG system across all test questions using all 4 prompt strategies.

Metrics computed:
  - Retrieval: precision@k (were expected papers retrieved?)
  - Generation: response length, token usage
  - Faithfulness (heuristic): keyword overlap between context and answer

Usage:
    python eval/evaluation_script.py                   # run all questions
    python eval/evaluation_script.py --questions 5     # first N questions
    python eval/evaluation_script.py --strategy cot    # single strategy
    python eval/evaluation_script.py --output results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of retrieved paper IDs that are in expected_ids."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for pid in retrieved_ids if pid in expected_ids)
    return round(hits / len(retrieved_ids), 4)


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of expected papers that were retrieved."""
    if not expected_ids:
        return 1.0
    hits = sum(1 for pid in expected_ids if pid in retrieved_ids)
    return round(hits / len(expected_ids), 4)


def keyword_overlap(answer: str, key_concepts: list[str]) -> float:
    """Fraction of key_concepts present (case-insensitive) in the answer."""
    if not key_concepts:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for kw in key_concepts if kw.lower() in answer_lower)
    return round(found / len(key_concepts), 4)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_question(
    question_record: dict,
    strategy: str,
    chunk_strategy: str = "small",
    top_k: int = 5,
    openai_client=None,
    chroma_client=None,
) -> dict:
    """Run the full RAG pipeline for one question/strategy and compute metrics."""
    from src.retrieval import search
    from src.generation import generate_answer

    q = question_record["question"]
    expected_papers = question_record.get("expected_papers", [])
    key_concepts = question_record.get("key_concepts", [])

    # Retrieval
    t0 = time.time()
    try:
        chunks = search(
            q,
            top_k=top_k,
            strategy=chunk_strategy,
            openai_client=openai_client,
            chroma_client=chroma_client,
        )
        retrieval_ok = True
    except RuntimeError as exc:
        logger.error(f"Retrieval failed: {exc}")
        return {"error": str(exc)}

    retrieval_time = round(time.time() - t0, 2)

    retrieved_paper_ids = list({c.paper_id for c in chunks})

    # Generation
    t1 = time.time()
    result = generate_answer(
        question=q,
        chunks=chunks,
        strategy=strategy,
        client=openai_client,
    )
    generation_time = round(time.time() - t1, 2)

    # Metrics
    p_at_k = precision_at_k(retrieved_paper_ids, expected_papers)
    r_at_k = recall_at_k(retrieved_paper_ids, expected_papers)
    kw_overlap = keyword_overlap(result["answer"], key_concepts)

    return {
        "question_id": question_record["id"],
        "category": question_record["category"],
        "difficulty": question_record["difficulty"],
        "strategy": strategy,
        "chunk_strategy": chunk_strategy,
        "top_k": top_k,
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "keyword_overlap": kw_overlap,
        "answer_length_chars": len(result["answer"]),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "retrieval_time_s": retrieval_time,
        "generation_time_s": generation_time,
        "answer_preview": result["answer"][:300],
        "retrieved_papers": retrieved_paper_ids,
        "expected_papers": expected_papers,
    }


def run_evaluation(
    questions: list[dict],
    strategies: list[str],
    chunk_strategy: str = "small",
    top_k: int = 5,
    output_path: Optional[Path] = None,
) -> dict:
    """Run full evaluation matrix: questions × strategies."""
    from openai import OpenAI
    from src.vectorstore import get_chroma_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    oa_client = OpenAI(api_key=api_key)
    chroma_client = get_chroma_client()

    all_results = []
    total = len(questions) * len(strategies)
    done = 0

    for q in questions:
        for strat in strategies:
            done += 1
            logger.info(
                f"[{done}/{total}] Q={q['id']} strategy={strat}"
            )
            res = evaluate_question(
                question_record=q,
                strategy=strat,
                chunk_strategy=chunk_strategy,
                top_k=top_k,
                openai_client=oa_client,
                chroma_client=chroma_client,
            )
            all_results.append(res)

    # Aggregate metrics per strategy
    summary: dict[str, dict] = {}
    for strat in strategies:
        strat_results = [r for r in all_results if r.get("strategy") == strat]
        if not strat_results:
            continue
        summary[strat] = {
            "n": len(strat_results),
            "avg_precision_at_k": round(
                sum(r["precision_at_k"] for r in strat_results) / len(strat_results), 4
            ),
            "avg_recall_at_k": round(
                sum(r["recall_at_k"] for r in strat_results) / len(strat_results), 4
            ),
            "avg_keyword_overlap": round(
                sum(r["keyword_overlap"] for r in strat_results) / len(strat_results), 4
            ),
            "avg_total_tokens": round(
                sum(r["total_tokens"] for r in strat_results) / len(strat_results)
            ),
            "avg_answer_length": round(
                sum(r["answer_length_chars"] for r in strat_results) / len(strat_results)
            ),
            "avg_generation_time_s": round(
                sum(r["generation_time_s"] for r in strat_results) / len(strat_results), 2
            ),
        }

    output = {
        "config": {
            "chunk_strategy": chunk_strategy,
            "top_k": top_k,
            "strategies": strategies,
            "n_questions": len(questions),
        },
        "summary": summary,
        "results": all_results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")

    return output


def print_summary(output: dict) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    summary = output["summary"]
    header = f"{'Strategy':<15} {'P@k':>7} {'R@k':>7} {'KW':>7} {'Tokens':>8} {'Time(s)':>8}"
    print(header)
    print("-" * 70)

    for strat, m in summary.items():
        print(
            f"{strat:<15} {m['avg_precision_at_k']:>7.3f} "
            f"{m['avg_recall_at_k']:>7.3f} "
            f"{m['avg_keyword_overlap']:>7.3f} "
            f"{m['avg_total_tokens']:>8.0f} "
            f"{m['avg_generation_time_s']:>8.2f}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["delimiters", "json", "fewshot", "cot"],
        help="Single strategy to evaluate (default: all 4)",
    )
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default="small",
        choices=["small", "large"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "eval" / "results.json"),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    # Load questions
    questions_path = ROOT / "eval" / "test_questions.json"
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    if args.questions:
        questions = questions[: args.questions]

    strategies = [args.strategy] if args.strategy else ["delimiters", "json", "fewshot", "cot"]

    logger.info(
        f"Evaluating {len(questions)} questions × {len(strategies)} strategies…"
    )

    output = run_evaluation(
        questions=questions,
        strategies=strategies,
        chunk_strategy=args.chunk_strategy,
        top_k=args.top_k,
        output_path=Path(args.output),
    )

    print_summary(output)
