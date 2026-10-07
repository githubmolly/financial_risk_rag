"""
Shared setup for project notebooks.

Usage in any notebook:
    from _setup import PROJECT_ROOT, retriever
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (parent of notebooks/)
_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here.parent if _here.name == "notebooks" else _here

# Make src.* importable
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv(PROJECT_ROOT / ".env", override=True)
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY not set — check .env")

# Build retriever once, reuse everywhere
from src.retrieval import HybridRetriever

retriever = HybridRetriever(
    chroma_path=str(PROJECT_ROOT / "chroma_db"),
    collection_name="financial_risk_10k",
)

print(f"✓ Setup complete | {len(retriever.chunk_texts)} chunks loaded")