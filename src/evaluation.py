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