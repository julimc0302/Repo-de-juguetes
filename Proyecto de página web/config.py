import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZES = [256, 1024]
CHUNK_OVERLAP = 50

CHROMA_PATH = "./chroma_db"
PDF_FOLDER = "."
CATALOG_PATH = "catalog.json"

TOP_K = 5
