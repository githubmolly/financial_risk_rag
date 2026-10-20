"""
src/evaluation.py — Evaluation framework for the Financial Risk RAG system

Two metric categories:
1. RETRIEVAL metrics (precision@k, recall@k, MRR) — does retrieval surface the right chunks?
2. GENERATION metrics (faithfulness, answer relevance via LLM-as-judge)
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from google import genai

from src.pipeline import RAGPipeline, RAGResponse
from src.retrieval import HybridRetriever


# ======================================================================
# Test set data structures
# ======================================================================
@dataclass
class TestQuestion:
    """One row of the golden test set."""
    id: str
    question: str
    expected_tickers: list[str]            # Which companies the answer should reference
    expected_keywords: list[str]           # Phrases that should appear in a correct answer
    reference_answer: str                  # Human-written ideal answer (for LLM-as-judge)
    category: str                          # e.g., "single_company", "comparison", "out_of_scope"
    should_refuse: bool = False            # Should the model refuse this question?


# ======================================================================
# Retrieval metrics
# ======================================================================
def precision_at_k(retrieved_tickers: list[str], expected_tickers: list[str], k: int) -> float:
    """Fraction of top-k retrieved chunks that come from expected companies."""
    if k == 0 or not retrieved_tickers:
        return 0.0
    top_k = retrieved_tickers[:k]
    relevant = sum(1 for t in top_k if t in expected_tickers)
    return relevant / k


def recall_at_k(retrieved_tickers: list[str], expected_tickers: list[str], k: int) -> float:
    """Fraction of expected companies that appear in top-k retrieved chunks."""
    if not expected_tickers:
        return 1.0  # vacuous
    top_k = set(retrieved_tickers[:k])
    found = sum(1 for t in expected_tickers if t in top_k)
    return found / len(expected_tickers)


def mean_reciprocal_rank(retrieved_tickers: list[str], expected_tickers: list[str]) -> float:
    """Reciprocal of rank of first relevant chunk."""
    for i, t in enumerate(retrieved_tickers, start=1):
        if t in expected_tickers:
            return 1.0 / i
    return 0.0


def keyword_recall_at_k(retrieved_texts: list[str], expected_keywords: list[str], k: int) -> float:
    """Fraction of expected keywords found (case-insensitive) in the top-k chunk texts.

    Ticker metrics only check *which company* a chunk came from. A reranker mostly
    reorders chunks *within* the same company's filing, so ticker-level P@5 can be
    flat even when the chunks got more on-topic. This is a cheap content-level proxy.
    """
    if not expected_keywords:
        return 1.0  # vacuous
    haystack = " ".join(retrieved_texts[:k]).lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in haystack)
    return found / len(expected_keywords)


# ======================================================================
# LLM-as-judge for generation quality
# ======================================================================
FAITHFULNESS_JUDGE_PROMPT = """You are evaluating whether an AI assistant's answer is FAITHFUL to the provided context.

A faithful answer makes claims that are directly supported by the context. An unfaithful answer makes claims that go beyond the context, contradict it, or invent information.

CONTEXT:
{context}

QUESTION: {question}

AI ANSWER:
{answer}

Evaluate the answer on faithfulness. Output ONLY a JSON object with this exact structure:
{{
  "faithfulness_score": <integer 1-5>,
  "reasoning": "<one sentence>",
  "unsupported_claims": ["<claim 1>", "<claim 2>"]
}}

Scoring rubric:
5 = Every claim is directly supported by the context
4 = One minor unsupported claim or slight extrapolation
3 = A few unsupported claims, but the core answer is supported
2 = Significant unsupported claims
1 = Mostly unsupported or contradicts the context

Output only the JSON, no other text."""


RELEVANCE_JUDGE_PROMPT = """You are evaluating whether an AI assistant's answer is RELEVANT to the user's question.

QUESTION: {question}
REFERENCE ANSWER: {reference_answer}
AI ANSWER: {answer}

Compare the AI answer to the reference answer. Output ONLY a JSON object:
{{
  "relevance_score": <integer 1-5>,
  "reasoning": "<one sentence>"
}}

Scoring rubric:
5 = Addresses all key points in the reference
4 = Addresses most key points
3 = Addresses some key points
2 = Mostly off-topic but touches the question
1 = Fails to address the question

Output only the JSON, no other text."""


class LLMJudge:
    """LLM-as-judge for faithfulness and relevance scoring."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name  # Use the bigger model for judging

    def _call_and_parse(self, prompt: str) -> dict:
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt,
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {"error": "parse_failure", "raw": text}

    def score_faithfulness(self, question: str, answer: str, context: str) -> dict:
        prompt = FAITHFULNESS_JUDGE_PROMPT.format(
            context=context, question=question, answer=answer,
        )
        return self._call_and_parse(prompt)

    def score_relevance(self, question: str, answer: str, reference_answer: str) -> dict:
        prompt = RELEVANCE_JUDGE_PROMPT.format(
            question=question, reference_answer=reference_answer, answer=answer,
        )
        return self._call_and_parse(prompt)


# ======================================================================
# End-to-end evaluator
# ======================================================================
@dataclass
class EvalResult:
    """All metrics for a single question."""
    question_id: str
    question: str
    category: str
    # Retrieval
    precision_at_5: float
    recall_at_5: float
    mrr: float
    retrieved_tickers: list[str]
    # Generation
    answer: str
    refused: bool
    refusal_correct: Optional[bool]  # None if not an out_of_scope question
    citation_hallucination: bool
    faithfulness_score: Optional[int]
    relevance_score: Optional[int]
    faithfulness_reasoning: str
    relevance_reasoning: str
    # Performance
    total_latency_ms: float


class Evaluator:
    """Runs a full evaluation pass over a test set."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        judge: LLMJudge,
        delay_seconds: float = 5.0,  # Rate limiting for Gemini free tier
    ):
        self.pipeline = pipeline
        self.judge = judge
        self.delay = delay_seconds

    def evaluate_one(self, q: TestQuestion) -> EvalResult:
        # Run the pipeline
        response = self.pipeline.query(q.question)
        retrieved_tickers = [
            c.metadata.get("ticker", "?") for c in response.retrieved_chunks
        ]

        # Retrieval metrics
        p5 = precision_at_k(retrieved_tickers, q.expected_tickers, k=5)
        r5 = recall_at_k(retrieved_tickers, q.expected_tickers, k=5)
        mrr = mean_reciprocal_rank(retrieved_tickers, q.expected_tickers)

        # Refusal correctness
        refusal_correct = None
        if q.should_refuse:
            refusal_correct = response.refused
        else:
            refusal_correct = not response.refused

        # LLM-as-judge (skip if refusal expected — judging refusal text doesn't make sense)
        faith_score = None
        faith_reason = ""
        rel_score = None
        rel_reason = ""

        if not q.should_refuse:
            context = "\n\n".join(
                f"[Chunk {i+1}] {c.text}" for i, c in enumerate(response.retrieved_chunks)
            )
            time.sleep(self.delay)
            faith = self.judge.score_faithfulness(q.question, response.answer, context)
            if "faithfulness_score" in faith:
                faith_score = faith["faithfulness_score"]
                faith_reason = faith.get("reasoning", "")

            time.sleep(self.delay)
            rel = self.judge.score_relevance(q.question, response.answer, q.reference_answer)
            if "relevance_score" in rel:
                rel_score = rel["relevance_score"]
                rel_reason = rel.get("reasoning", "")

        return EvalResult(
            question_id=q.id,
            question=q.question,
            category=q.category,
            precision_at_5=p5,
            recall_at_5=r5,
            mrr=mrr,
            retrieved_tickers=retrieved_tickers,
            answer=response.answer,
            refused=response.refused,
            refusal_correct=refusal_correct,
            citation_hallucination=response.citation_hallucination,
            faithfulness_score=faith_score,
            relevance_score=rel_score,
            faithfulness_reasoning=faith_reason,
            relevance_reasoning=rel_reason,
            total_latency_ms=response.total_latency_ms,
        )

    def evaluate_all(self, questions: list[TestQuestion]) -> list[EvalResult]:
        results = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q.id}: {q.question[:60]}...")
            try:
                result = self.evaluate_one(q)
                results.append(result)
                print(f"  P@5={result.precision_at_5:.2f}  R@5={result.recall_at_5:.2f}  "
                      f"faith={result.faithfulness_score}  rel={result.relevance_score}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            time.sleep(self.delay)  # space out between questions
        return results


# ======================================================================
# Summary statistics
# ======================================================================
def summarize_results(results: list[EvalResult]) -> dict:
    """Aggregate metrics across the test set."""
    n = len(results)
    if n == 0:
        return {}

    # Retrieval aggregates
    avg_p5 = sum(r.precision_at_5 for r in results) / n
    avg_r5 = sum(r.recall_at_5 for r in results) / n
    avg_mrr = sum(r.mrr for r in results) / n

    # Generation aggregates (skip None scores)
    faith_scores = [r.faithfulness_score for r in results if r.faithfulness_score is not None]
    rel_scores = [r.relevance_score for r in results if r.relevance_score is not None]
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else None
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else None

    # Quality flags
    n_hallucinations = sum(1 for r in results if r.citation_hallucination)
    n_refusal_correct = sum(1 for r in results if r.refusal_correct)
    n_refusal_evaluated = sum(1 for r in results if r.refusal_correct is not None)

    return {
        "n_questions": n,
        "retrieval": {
            "avg_precision_at_5": avg_p5,
            "avg_recall_at_5": avg_r5,
            "avg_mrr": avg_mrr,
        },
        "generation": {
            "avg_faithfulness": avg_faith,
            "avg_relevance": avg_rel,
            "citation_hallucinations": n_hallucinations,
            "refusal_accuracy": (n_refusal_correct / n_refusal_evaluated) if n_refusal_evaluated else None,
        },
        "avg_latency_ms": sum(r.total_latency_ms for r in results) / n,
    }


def save_results(results: list[EvalResult], path: str) -> None:
    """Persist evaluation results to JSON for diff-against-previous-runs."""
    data = [asdict(r) for r in results]
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"✓ Saved {len(results)} results to {path}")


# ======================================================================
# Retrieval-only A/B: baseline (RRF) vs. cross-encoder reranking
# ======================================================================
# The reranker only changes retrieval, so compare it at the retrieval layer first:
# no Gemini calls means no rate limits, no judge noise, and a deterministic result.
# Run the full Evaluator on both arms only once this shows a retrieval win.
@dataclass
class RetrievalResult:
    """Retrieval-only metrics for one question under one retrieval config."""
    question_id: str
    category: str
    retrieved_ids: list[str]
    retrieved_tickers: list[str]
    precision_at_5: float
    recall_at_5: float
    mrr: float
    keyword_recall_at_5: float
    latency_ms: float


def evaluate_retrieval(
    retriever: HybridRetriever,
    questions: list[TestQuestion],
    use_reranker: bool,
    top_k: int = 5,
    candidate_pool: int = 25,
) -> list[RetrievalResult]:
    """Run retrieval only (no generation) over the test set."""
    results = []
    for q in questions:
        t0 = time.time()
        chunks = retriever.hybrid_search(
            query=q.question,
            top_k=top_k,
            candidate_pool=candidate_pool,
            use_reranker=use_reranker,
        )
        latency_ms = (time.time() - t0) * 1000

        tickers = [c.metadata.get("ticker", "?") for c in chunks]
        texts = [c.text for c in chunks]
        results.append(RetrievalResult(
            question_id=q.id,
            category=q.category,
            retrieved_ids=[c.chunk_id for c in chunks],
            retrieved_tickers=tickers,
            precision_at_5=precision_at_k(tickers, q.expected_tickers, k=5),
            recall_at_5=recall_at_k(tickers, q.expected_tickers, k=5),
            mrr=mean_reciprocal_rank(tickers, q.expected_tickers),
            keyword_recall_at_5=keyword_recall_at_k(texts, q.expected_keywords, k=5),
            latency_ms=latency_ms,
        ))
    return results


def summarize_retrieval(results: list[RetrievalResult]) -> dict:
    """Same shape as summarize_results() so print_comparison() works on both."""
    n = len(results)
    if n == 0:
        return {}
    return {
        "n_questions": n,
        "retrieval": {
            "avg_precision_at_5": sum(r.precision_at_5 for r in results) / n,
            "avg_recall_at_5": sum(r.recall_at_5 for r in results) / n,
            "avg_mrr": sum(r.mrr for r in results) / n,
            "avg_keyword_recall_at_5": sum(r.keyword_recall_at_5 for r in results) / n,
        },
        "avg_latency_ms": sum(r.latency_ms for r in results) / n,
    }


def print_comparison(
    summary_a: dict,
    summary_b: dict,
    label_a: str = "baseline",
    label_b: str = "reranked",
) -> None:
    """Side-by-side table of two summaries (from summarize_results or summarize_retrieval)."""
    def fmt(v):
        return f"{v:.3f}" if v is not None else "—"

    rows = []
    for section in ("retrieval", "generation"):
        for metric in summary_a.get(section, {}):
            rows.append((metric, summary_a[section][metric], summary_b.get(section, {}).get(metric)))
    rows.append(("avg_latency_ms", summary_a.get("avg_latency_ms"), summary_b.get("avg_latency_ms")))

    print(f"\n{'Metric':<28} {label_a:<14} {label_b:<14} {'Δ':<10}")
    print("-" * 66)
    for metric, va, vb in rows:
        numeric = isinstance(va, (int, float)) and isinstance(vb, (int, float))
        delta = f"{vb - va:+.3f}" if numeric else "—"
        print(f"{metric:<28} {fmt(va):<14} {fmt(vb):<14} {delta:<10}")


def compare_rerank_retrieval(
    retriever: HybridRetriever,
    questions: list[TestQuestion],
    top_k: int = 5,
    candidate_pool: int = 25,
) -> dict:
    """Retrieval-only A/B of RRF baseline vs. RRF + cross-encoder rerank.

    Out-of-scope (should_refuse) questions are excluded: they have no correct
    chunks, so retrieval metrics on them are meaningless and only dilute averages.
    """
    in_scope = [q for q in questions if not q.should_refuse]
    print(f"Comparing on {len(in_scope)} in-scope questions "
          f"({len(questions) - len(in_scope)} out-of-scope excluded)")

    # Warm-up: the first rerank call loads the cross-encoder (and may download it).
    # Without this, model load time lands in the first question's latency.
    retriever.hybrid_search(in_scope[0].question, top_k=top_k,
                            candidate_pool=candidate_pool, use_reranker=True)

    baseline = evaluate_retrieval(retriever, in_scope, use_reranker=False,
                                  top_k=top_k, candidate_pool=candidate_pool)
    reranked = evaluate_retrieval(retriever, in_scope, use_reranker=True,
                                  top_k=top_k, candidate_pool=candidate_pool)

    summary_baseline = summarize_retrieval(baseline)
    summary_reranked = summarize_retrieval(reranked)
    print_comparison(summary_baseline, summary_reranked)

    # Per-question view: an average can hide a reranker that helps some questions
    # and hurts others. Overlap shows how much the reranker actually changed.
    print(f"\n{'Question':<16} {'Category':<16} {'Overlap':<9} {'ΔP@5':<8} {'ΔMRR':<8} {'ΔKW@5':<8}")
    print("-" * 66)
    for b, r in zip(baseline, reranked):
        overlap = len(set(b.retrieved_ids) & set(r.retrieved_ids))
        print(f"{b.question_id:<16} {b.category:<16} {overlap}/{top_k:<7} "
              f"{r.precision_at_5 - b.precision_at_5:<+8.2f} "
              f"{r.mrr - b.mrr:<+8.2f} "
              f"{r.keyword_recall_at_5 - b.keyword_recall_at_5:<+8.2f}")

    return {
        "baseline": baseline,
        "reranked": reranked,
        "summary_baseline": summary_baseline,
        "summary_reranked": summary_reranked,
    }


def save_retrieval_comparison(comparison: dict, path: str) -> None:
    """Persist a compare_rerank_retrieval() result to JSON."""
    data = {
        "summary_baseline": comparison["summary_baseline"],
        "summary_reranked": comparison["summary_reranked"],
        "baseline": [asdict(r) for r in comparison["baseline"]],
        "reranked": [asdict(r) for r in comparison["reranked"]],
    }
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"✓ Saved retrieval comparison to {path}")
