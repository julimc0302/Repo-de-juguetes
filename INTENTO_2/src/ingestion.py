"""
ingestion.py
------------
Extracts text from PDFs using PyMuPDF and combines it with metadata
from papers.json. Running this module directly indexes all papers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Default paths (relative to project root)
PAPERS_DIR = Path(__file__).resolve().parent.parent / "papers"
PAPERS_JSON = PAPERS_DIR / "papers.json"


def load_metadata(json_path: Path = PAPERS_JSON) -> dict[str, dict]:
    """Return a dict keyed by filename → metadata record."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["filename"]: p for p in data["papers"]}


def extract_text(pdf_path: Path) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n".join(pages)


def _find_metadata(fname: str, metadata: dict[str, dict]) -> Optional[dict]:
    """
    Find metadata for a filename using exact match first, then fuzzy prefix match.

    Handles two common mismatches:
      1. JSON filenames are truncated (e.g. 'Author - 2020 - Title....pdf')
         while actual files have the full name.
      2. Actual files may lack the .pdf extension.
    """
    # 1. Exact match
    if fname in metadata:
        return metadata[fname]

    # 2. Try without extension (file has no .pdf)
    fname_noext = fname if not fname.endswith(".pdf") else fname[:-4]

    for key, meta in metadata.items():
        key_noext = key[:-4] if key.endswith(".pdf") else key
        # Remove trailing ellipsis from JSON key if present
        key_noext = key_noext.rstrip(".")

        # Match if either name starts with a common prefix (first 30 chars)
        prefix_len = min(30, len(fname_noext), len(key_noext))
        if prefix_len >= 15 and fname_noext[:prefix_len] == key_noext[:prefix_len]:
            return meta

    return None


def _find_paper_files(papers_dir: Path) -> list[Path]:
    """
    Return all PDF files in papers_dir.
    Uses PyMuPDF to detect valid PDFs, handling:
      - Files without .pdf extension (e.g. pathlib misparses 'Author et al. - ...')
      - Files with non-standard headers (BOM, CR/LF before %PDF-)
    """
    SKIP_NAMES = {"papers.json"}
    SKIP_EXTS = {".json", ".txt", ".md", ".py", ".ipynb"}
    files = []
    for path in sorted(papers_dir.iterdir()):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        if any(path.name.lower().endswith(ext) for ext in SKIP_EXTS):
            continue
        # Try opening with PyMuPDF — most reliable PDF detection
        try:
            doc = fitz.open(str(path))
            if doc.page_count > 0:
                files.append(path)
            doc.close()
        except Exception:
            pass
    return files


def load_papers(
    papers_dir: Path = PAPERS_DIR,
    json_path: Path = PAPERS_JSON,
    verbose: bool = True,
) -> list[dict]:
    """
    Load all papers, extracting text and merging metadata.

    Returns a list of dicts, each with:
        id, title, authors, year, venue, doi, topics, abstract,
        filename, text, num_chars, num_pages
    """
    metadata = load_metadata(json_path)
    results = []

    paper_files = _find_paper_files(papers_dir)
    if verbose:
        logger.info(f"Found {len(paper_files)} paper files in {papers_dir}")

    for pdf_path in paper_files:
        fname = pdf_path.name
        meta = _find_metadata(fname, metadata)

        if meta is None:
            logger.warning(f"No metadata found for: {fname} — skipping")
            continue

        try:
            text = extract_text(pdf_path)
        except Exception as exc:
            logger.error(f"Failed to extract text from {fname}: {exc}")
            continue

        doc = fitz.open(str(pdf_path))
        num_pages = doc.page_count
        doc.close()

        record = {
            **meta,
            "text": text,
            "num_chars": len(text),
            "num_pages": num_pages,
            "filepath": str(pdf_path),
        }
        results.append(record)

        if verbose:
            logger.info(
                f"  [{meta['id']}] {meta['title'][:60]} — "
                f"{num_pages} pages, {len(text):,} chars"
            )

    if verbose:
        logger.info(f"Loaded {len(results)} papers successfully.")
    return results


def get_paper_by_id(paper_id: str, papers: list[dict]) -> Optional[dict]:
    """Return the paper dict matching the given id, or None."""
    for p in papers:
        if p.get("id") == paper_id:
            return p
    return None


# ---------------------------------------------------------------------------
# CLI entry-point: python -m src.ingestion
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    papers = load_papers(verbose=True)
    print(f"\nSummary: {len(papers)} papers loaded.")
    total_chars = sum(p["num_chars"] for p in papers)
    print(f"Total text: {total_chars:,} characters across all papers.")
