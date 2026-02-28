"""
ingest.py — Extract text from PDFs, chunk, embed, and store in ChromaDB.

Usage:
    python ingest.py
"""

import os
import json
import re
import sys
import hashlib
from pathlib import Path

import pdfplumber
import tiktoken
import chromadb
import openai
from dotenv import load_dotenv

# ── Load config ──────────────────────────────────────────────────────────────
load_dotenv()

# Import from config when running as part of the package; fall back to defaults
try:
    from config import (
        OPENAI_API_KEY, EMBEDDING_MODEL, CHUNK_SIZES, CHUNK_OVERLAP,
        CHROMA_PATH, PDF_FOLDER, CATALOG_PATH,
    )
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHUNK_SIZES = [256, 1024]
    CHUNK_OVERLAP = 50
    CHROMA_PATH = "./chroma_db"
    PDF_FOLDER = "."
    CATALOG_PATH = "catalog.json"

openai.api_key = OPENAI_API_KEY

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> dict:
    """
    Extract full text and metadata from a PDF file.

    Returns a dict with keys: text, title, authors, year, pages.
    """
    result = {
        "text": "",
        "title": Path(path).stem,
        "authors": "Unknown",
        "year": None,
        "pages": 0,
    }
    pages_text = []
    try:
        with pdfplumber.open(path) as pdf:
            result["pages"] = len(pdf.pages)

            # Try PDF metadata first
            meta = pdf.metadata or {}
            if meta.get("Title"):
                result["title"] = meta["Title"].strip()
            if meta.get("Author"):
                result["authors"] = meta["Author"].strip()
            if meta.get("CreationDate"):
                year_match = re.search(r"(\d{4})", meta["CreationDate"])
                if year_match:
                    result["year"] = int(year_match.group(1))

            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)

    except Exception as exc:
        print(f"  [WARN] pdfplumber error on {path}: {exc}", file=sys.stderr)

    full_text = "\n".join(pages_text)
    result["text"] = full_text

    # Fallback: try to parse year from first page text
    if result["year"] is None and pages_text:
        year_match = re.search(r"\b(19|20)\d{2}\b", pages_text[0])
        if year_match:
            result["year"] = int(year_match.group())

    if result["year"] is None:
        result["year"] = 0  # unknown

    return result


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of `chunk_size` tokens with `overlap` token overlap.
    Uses tiktoken cl100k_base encoding.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        if chunk_text_str.strip():
            chunks.append(chunk_text_str)
        if end >= len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def get_or_create_collection(client: chromadb.Client, name: str) -> chromadb.Collection:
    """Return existing ChromaDB collection or create it."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using OpenAI embeddings.
    Batches in groups of 100 to stay within API limits.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def _make_chunk_id(filename: str, chunk_size: int, chunk_index: int) -> str:
    """Stable unique ID for a chunk."""
    raw = f"{filename}_{chunk_size}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_all_pdfs() -> list[dict]:
    """
    Main ingestion loop:
    1. Scan PDF_FOLDER for .pdf files.
    2. Extract text + metadata.
    3. Chunk at two sizes (256, 1024).
    4. Embed and upsert into ChromaDB collections.
    5. Save catalog.json.

    Returns the catalog list.
    """
    pdf_folder = Path(PDF_FOLDER)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{pdf_folder.resolve()}'")
        return []

    print(f"Found {len(pdf_files)} PDF(s) in '{pdf_folder.resolve()}'")

    # Initialise ChromaDB (persistent)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = {
        cs: get_or_create_collection(chroma_client, f"papers_{cs}")
        for cs in CHUNK_SIZES
    }

    catalog = []

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        extracted = extract_text_from_pdf(str(pdf_path))

        if not extracted["text"].strip():
            print(f"  [WARN] No text extracted from {pdf_path.name}, skipping.")
            continue

        entry = {
            "filename": pdf_path.name,
            "title": extracted["title"],
            "authors": extracted["authors"],
            "year": extracted["year"],
            "pages": extracted["pages"],
        }

        for chunk_size in CHUNK_SIZES:
            chunks = chunk_text(extracted["text"], chunk_size, CHUNK_OVERLAP)
            print(f"  Chunk size {chunk_size}: {len(chunks)} chunks")

            if not chunks:
                entry[f"chunk_count_{chunk_size}"] = 0
                continue

            entry[f"chunk_count_{chunk_size}"] = len(chunks)

            # Build IDs and metadata
            ids = [_make_chunk_id(pdf_path.name, chunk_size, i) for i in range(len(chunks))]
            metadatas = [
                {
                    "filename": pdf_path.name,
                    "title": extracted["title"],
                    "authors": extracted["authors"],
                    "year": extracted["year"],
                    "chunk_index": i,
                    "chunk_size": chunk_size,
                    "page_start": 1,  # granular page tracking not implemented
                }
                for i in range(len(chunks))
            ]

            print(f"  Embedding {len(chunks)} chunks (size={chunk_size})…")
            embeddings = embed_texts(chunks)

            collections[chunk_size].upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        catalog.append(entry)

    # Save catalog
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"\nCatalog saved to '{CATALOG_PATH}' with {len(catalog)} entries.")
    print(f"ChromaDB stored at '{CHROMA_PATH}'.")
    return catalog


if __name__ == "__main__":
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-..."):
        print("ERROR: Set OPENAI_API_KEY in your .env file before running ingest.")
        sys.exit(1)
    ingest_all_pdfs()
