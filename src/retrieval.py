# query -> relevant chunks
"""
src/retrieval.py — Hybrid retrieval for the Financial Risk RAG system

Combines ChromaDB dense search with BM25 keyword search using
Reciprocal Rank Fusion. Supports metadata filtering (e.g., by ticker).
"""

import re
from typing import Optional
from dataclasses import dataclass

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from typing import Optional

from src.reranker import get_reranker

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with all relevant context."""
    chunk_id: str           # ChromaDB document id
    text: str               # The chunk text
    metadata: dict          # ticker, accession, chunk_index, etc.
    score: float            # Combined score (RRF)
    dense_rank: int         # Rank in dense search (-1 if not in top results)
    bm25_rank: int          # Rank in BM25 search (-1 if not in top results)


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class HybridRetriever:
    """Hybrid retrieval over a ChromaDB collection using dense + BM25."""

    def __init__(
        self,
        chroma_path: str,
        collection_name: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)

        # Load embedding model (same as used at ingestion)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Pull all chunks once to build BM25 index in memory
        print(f"Loading chunks from collection '{collection_name}'...")
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.chunk_ids: list[str] = all_data["ids"]
        self.chunk_texts: list[str] = all_data["documents"]
        self.chunk_metas: list[dict] = all_data["metadatas"]

        # Map id -> index for fast lookup
        self.id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}

        # Build BM25 index
        print(f"Building BM25 index over {len(self.chunk_texts)} chunks...")
        tokenized = [_tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

        print(f"✓ HybridRetriever ready: {len(self.chunk_texts)} chunks")

    # ------------------------------------------------------------------
    # Individual search methods (useful for debugging / ablation studies)
    # ------------------------------------------------------------------
    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Pure dense (ChromaDB) retrieval, optionally with metadata filter."""
        query_embedding = self.embedding_model.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,  # e.g., {"ticker": "JPM"}
        )

        # ChromaDB returns lists wrapped in another list (for batch queries)
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # ChromaDB returns distance; convert to similarity (cosine: 1 - distance)
        return [
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                metadata=meta,
                score=1.0 - dist,
                dense_rank=rank,
                bm25_rank=-1,
            )
            for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances))
        ]

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Pure BM25 retrieval, optionally with metadata filter."""
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Apply metadata filter if provided
        if where:
            mask = np.array([
                all(meta.get(k) == v for k, v in where.items())
                for meta in self.chunk_metas
            ])
            scores = np.where(mask, scores, -np.inf)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedChunk(
                chunk_id=self.chunk_ids[idx],
                text=self.chunk_texts[idx],
                metadata=self.chunk_metas[idx],
                score=float(scores[idx]),
                dense_rank=-1,
                bm25_rank=rank,
            )
            for rank, idx in enumerate(top_indices)
            if scores[idx] > -np.inf
        ]

    # ------------------------------------------------------------------
    # The main one: hybrid via RRF
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 25,
        where: Optional[dict] = None,
        rrf_k: int = 60,
        use_reranker: bool = False,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval (BM25 + dense via RRF) with optional cross-encoder reranking.

        Two-stage retrieval:
          - Stage 1: Pull `candidate_pool` chunks from each of dense (ChromaDB)
            and BM25 search, fuse via Reciprocal Rank Fusion.
          - Stage 2 (optional): Rescore top candidates with a cross-encoder
            reranker, which attends across (query, chunk) jointly rather than
            comparing precomputed vectors.

        Args:
            query: User's question.
            top_k: Final number of chunks to return to the generator.
            candidate_pool: How many chunks to pull from each first-stage retriever
                before fusion. Larger pool = more candidates for the reranker to
                choose from, at the cost of reranker latency.
            where: Optional metadata filter passed to both retrievers (e.g.,
                {"ticker": "JPM"}).
            rrf_k: Reciprocal Rank Fusion smoothing constant. Standard value is 60.
            use_reranker: If True, apply cross-encoder reranking after RRF fusion.
                Defaults to False for backward compatibility.

        Returns:
            Up to top_k chunks, sorted by RRF score (if use_reranker=False) or
            by cross-encoder rerank score (if use_reranker=True). Each chunk's
            .score field reflects whichever ranking was used.
        """
        # --- Stage 1: dense + BM25 retrieval ---
        dense_results = self.dense_search(query, top_k=candidate_pool, where=where)
        bm25_results = self.bm25_search(query, top_k=candidate_pool, where=where)

        # Build RRF score per chunk_id
        rrf_scores: dict[str, float] = {}
        chunk_info: dict[str, RetrievedChunk] = {}

        for chunk in dense_results:
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (rrf_k + chunk.dense_rank)
            chunk_info[chunk.chunk_id] = chunk

        for chunk in bm25_results:
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (rrf_k + chunk.bm25_rank)
            if chunk.chunk_id in chunk_info:
                chunk_info[chunk.chunk_id].bm25_rank = chunk.bm25_rank
            else:
                chunk_info[chunk.chunk_id] = chunk

        # --- Determine how many to carry forward from Stage 1 ---
        # When reranking, keep the full fused candidate set so the cross-encoder
        # has material to choose from. When not reranking, truncate to top_k now.
        stage_1_cutoff = len(rrf_scores) if use_reranker else top_k

        ranked_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:stage_1_cutoff]

        stage_1_results = []
        for cid in ranked_ids:
            c = chunk_info[cid]
            c.score = rrf_scores[cid]  # RRF-based score
            stage_1_results.append(c)

        # --- Stage 2 (optional): cross-encoder reranking ---
        if not use_reranker:
            return stage_1_results

        reranker = get_reranker()
        reranked = reranker.rerank(query, stage_1_results, top_k=top_k)
        return reranked
