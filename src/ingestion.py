import json
import time
from itertools import groupby
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk, run_chunking

COLLECTION_NAME = "financial_risk_10k"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_or_create_collection(chroma_path: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _already_ingested(collection: chromadb.Collection, source: str) -> bool:
    """File-level idempotency: True if any chunk from this source exists."""
    result = collection.get(where={"source": source}, limit=1, include=[])
    return len(result["ids"]) > 0


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Core ingest loop
# ---------------------------------------------------------------------------

def ingest_chunks(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> dict:
    """
    Ingest chunks into ChromaDB in batches of BATCH_SIZE.

    Idempotency is enforced at the file level: if the collection already
    contains any chunk from a given source file, the entire file is skipped.
    Chunks are grouped and sorted by source before processing so the groupby
    sees contiguous runs.
    """
    added = 0
    skipped_sources: list[str] = []

    by_source = groupby(
        sorted(chunks, key=lambda c: c.source),
        key=lambda c: c.source,
    )

    for source, source_iter in by_source:
        source_chunks = list(source_iter)

        if _already_ingested(collection, source):
            print(f"  skip  {source}  (already in collection)")
            skipped_sources.append(source)
            continue

        n = len(source_chunks)
        print(f"  add   {source}  ({n} chunks) ", end="", flush=True)

        for batch_start in range(0, n, BATCH_SIZE):
            batch = source_chunks[batch_start : batch_start + BATCH_SIZE]
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=_embed(model, [c.text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[c.to_chroma_metadata() for c in batch],
            )
            print(".", end="", flush=True)
            added += len(batch)

        print()  # newline after progress dots

    return {
        "added": added,
        "skipped_sources": skipped_sources,
        "total_in_collection": collection.count(),
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_ingestion(
    chunks: list[Chunk] | None = None,
    extracted_dir: str | Path = "data/extracted_text",
    chunks_dir: str | Path = "data/chunks",
    chroma_path: str | Path = "chroma_db",
) -> dict:
    """
    Main entry point.  Accepts pre-built chunks or loads from chunks.json.
    Falls back to running the chunking pipeline if neither is available.
    """
    t0 = time.time()

    # ── resolve chunks ──────────────────────────────────────────────────────
    if chunks is None:
        chunks_file = Path(chunks_dir) / "chunks.json"
        if chunks_file.exists():
            print(f"Loading chunks from {chunks_file} ...")
            raw = json.loads(chunks_file.read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in raw]
            print(f"  {len(chunks)} chunks loaded")
        else:
            print("chunks.json not found — running chunking pipeline first ...")
            chunks = run_chunking(extracted_dir, chunks_dir)

    # ── embedding model ─────────────────────────────────────────────────────
    print(f"\nLoading embedding model  {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    # ── collection ──────────────────────────────────────────────────────────
    chroma_path = Path(chroma_path)
    collection = _get_or_create_collection(chroma_path)
    print(f"Collection '{COLLECTION_NAME}'  —  {collection.count()} existing chunks\n")

    # ── ingest ──────────────────────────────────────────────────────────────
    stats = ingest_chunks(chunks, collection, model)
    stats["elapsed_s"] = round(time.time() - t0, 1)

    print(
        f"\nDone — "
        f"added={stats['added']}  "
        f"skipped={len(stats['skipped_sources'])}  "
        f"total_in_db={stats['total_in_collection']}  "
        f"time={stats['elapsed_s']}s"
    )
    return stats


if __name__ == "__main__":
    import sys
    _chroma = sys.argv[1] if len(sys.argv) > 1 else "chroma_db"
    run_ingestion(chroma_path=_chroma)
