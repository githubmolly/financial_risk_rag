# ties extraction, chunking, ingestion, retrieval, and generation together
"""
src/pipeline.py — Top-level RAG pipeline orchestrator
"""

import time
from dataclasses import dataclass
from typing import Optional

from src.retrieval import HybridRetriever, RetrievedChunk
from src.generation import Generator, GenerationResult


@dataclass
class RAGResponse:
    """The complete output of a single query through the pipeline."""
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    cited_chunk_numbers: list[int]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    # Quality diagnostics
    citation_hallucination: bool  # True if model cited chunks that don't exist
    refused: bool                  # True if model refused to answer


class RAGPipeline:
    """End-to-end RAG: retrieval + generation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        top_k: int = 5,
        candidate_pool: int = 25,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.candidate_pool = candidate_pool

    def query(
        self,
        question: str,
        where: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """Execute the full RAG pipeline for a single question."""
        top_k = top_k or self.top_k

        # ---- Retrieval ----
        t0 = time.time()
        chunks = self.retriever.hybrid_search(
            query=question,
            top_k=top_k,
            candidate_pool=self.candidate_pool,
            where=where,
        )
        retrieval_ms = (time.time() - t0) * 1000

        # ---- Generation ----
        t1 = time.time()
        gen_result = self.generator.generate(question=question, chunks=chunks)
        generation_ms = (time.time() - t1) * 1000

        # ---- Diagnostics ----
        n_chunks = len(chunks)
        cited = gen_result.cited_chunk_numbers
        citation_hallucination = any(c < 1 or c > n_chunks for c in cited)
        refused = "do not contain sufficient information" in gen_result.answer.lower()

        return RAGResponse(
            question=question,
            answer=gen_result.answer,
            retrieved_chunks=chunks,
            cited_chunk_numbers=cited,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            total_latency_ms=retrieval_ms + generation_ms,
            citation_hallucination=citation_hallucination,
            refused=refused,
        )

    def pretty_print(self, response: RAGResponse) -> None:
        """Pretty print a response for interactive use."""
        print("=" * 80)
        print(f"QUESTION: {response.question}")
        print("=" * 80)
        print(f"\nANSWER:\n{response.answer}\n")
        print(f"Cited chunks: {response.cited_chunk_numbers}")
        print(f"Latency: retrieval={response.retrieval_latency_ms:.0f}ms, "
              f"generation={response.generation_latency_ms:.0f}ms")
        if response.citation_hallucination:
            print("⚠️ CITATION HALLUCINATION DETECTED")
        if response.refused:
            print("ℹ️ Model refused to answer (insufficient context)")
        print("\n--- RETRIEVED CHUNKS ---")
        for i, chunk in enumerate(response.retrieved_chunks, start=1):
            ticker = chunk.metadata.get("ticker", "?")
            marker = "✓" if i in response.cited_chunk_numbers else " "
            print(f"\n{marker} [Chunk {i}] {ticker} (RRF score: {chunk.score:.4f})")
            print(f"   {chunk.text[:200]}...")