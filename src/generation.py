"""
src/generation.py — Grounded answer generation with versioned prompts

Supports three providers: anthropic (Claude), gemini, deepseek.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from src.retrieval import RetrievedChunk


# ======================================================================
# Versioned prompt templates (unchanged)
# ======================================================================
PROMPT_V1_BASELINE = """You are a financial risk analyst answering questions about SEC 10-K filings.

You will be given excerpts from the "Risk Factors" sections of multiple companies' annual reports. Answer the user's question using ONLY these excerpts. Follow these rules strictly:

RULES:
1. Base every claim on the provided context. Do NOT use outside knowledge.
2. If the context does not contain enough information to answer, say exactly:
   "The provided excerpts do not contain sufficient information to answer this question."
   Do NOT speculate or guess.
3. When you make a factual claim, cite the source using [Chunk N] notation where N is the chunk number.
4. When comparing companies, structure your answer by company.
5. Do NOT cite chunks that aren't provided. Only chunks 1 through {n_chunks} exist for this query.
6. Be concise but complete. Avoid filler language.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (cite every claim with [Chunk N]):"""


PROMPT_V2_ENFORCED_SHAPE = """You are a financial risk analyst answering questions about SEC 10-K filings.

You will be given excerpts from the "Risk Factors" sections of multiple companies' annual reports. Answer the user's question using ONLY these excerpts.

RULES:
1. Base every claim on the provided context. Do NOT use outside knowledge.
2. If the context does not contain enough information to answer, say exactly:
   "The provided excerpts do not contain sufficient information to answer this question."
3. Cite every factual claim using [Chunk N] notation.
4. Do NOT cite chunks that aren't provided. Only chunks 1 through {n_chunks} exist.
5. Be concise but complete.

CRITICAL — ANSWER SHAPE:
- If the question asks how multiple companies DIFFER, COMPARE, or CONTRAST, you MUST:
  a) Identify each company mentioned in the question
  b) Write a separate labeled section for each (e.g., "**JPM:** ...", "**GS:** ...")
  c) End with a "**Key differences:**" section that explicitly contrasts them
- If the question asks about a single company, answer in flowing prose.
- If the question asks "which filings mention X", list each company that does, with one supporting quote.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (follow the answer shape rules above, cite with [Chunk N]):"""


PROMPT_VERSIONS = {
    "v1_baseline": PROMPT_V1_BASELINE,
    "v2_enforced_shape": PROMPT_V2_ENFORCED_SHAPE,
}


@dataclass
class GenerationResult:
    answer: str
    question: str
    chunks_provided: list[RetrievedChunk]
    cited_chunk_numbers: list[int] = field(default_factory=list)
    prompt_version: str = ""
    provider: str = ""
    raw_response: Optional[object] = None


class Generator:
    """Wraps Claude / Gemini / DeepSeek with a selectable prompt version."""

    def __init__(
        self,
        provider: str = "anthropic",   # NEW default: Claude
        model_name: Optional[str] = None,
        prompt_version: str = "v2_enforced_shape",
        api_key: Optional[str] = None,
    ):
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(
                f"Unknown prompt_version '{prompt_version}'. "
                f"Available: {list(PROMPT_VERSIONS.keys())}"
            )
        self.prompt_version = prompt_version
        self.prompt_template = PROMPT_VERSIONS[prompt_version]
        self.provider = provider

        if provider == "anthropic":
            self.model_name = model_name or "claude-haiku-4-5-20251001"
            self._init_anthropic(api_key)
        elif provider == "gemini":
            self.model_name = model_name or "gemini-2.5-flash-lite"
            self._init_gemini(api_key)
        elif provider == "deepseek":
            self.model_name = model_name or "deepseek-v4-flash"
            self._init_deepseek(api_key)
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Use 'anthropic', 'gemini', or 'deepseek'."
            )

    # --- provider-specific client initialization ---
    def _init_anthropic(self, api_key: Optional[str]):
        from anthropic import Anthropic
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment or passed in")
        self.client = Anthropic(api_key=api_key)

    def _init_gemini(self, api_key: Optional[str]):
        from google import genai
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment or passed in")
        self.client = genai.Client(api_key=api_key)

    def _init_deepseek(self, api_key: Optional[str]):
        from openai import OpenAI
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set in environment or passed in")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # --- main entry point ---
    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            ticker = chunk.metadata.get("ticker", "UNKNOWN")
            context_parts.append(
                f"[Chunk {i}] (Source: {ticker} 10-K Risk Factors)\n{chunk.text}"
            )
        context = "\n\n".join(context_parts)

        prompt = self.prompt_template.format(
            n_chunks=len(chunks),
            context=context,
            question=question,
        )

        if self.provider == "anthropic":
            answer_text, raw = self._generate_anthropic(prompt)
        elif self.provider == "gemini":
            answer_text, raw = self._generate_gemini(prompt)
        elif self.provider == "deepseek":
            answer_text, raw = self._generate_deepseek(prompt)
        else:
            raise RuntimeError(f"Unknown provider {self.provider}")

        cited = sorted(set(int(n) for n in re.findall(r"\[Chunk (\d+)\]", answer_text)))

        return GenerationResult(
            answer=answer_text,
            question=question,
            chunks_provided=chunks,
            cited_chunk_numbers=cited,
            prompt_version=self.prompt_version,
            provider=self.provider,
            raw_response=raw,
        )

    def _generate_anthropic(self, prompt: str):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        # Claude returns a list of content blocks; concatenate text blocks
        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return answer_text, response

    def _generate_gemini(self, prompt: str):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text, response

    def _generate_deepseek(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content, response