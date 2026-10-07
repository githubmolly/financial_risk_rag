# Financial Risk RAG — Retrieval-Augmented Q&A over SEC 10-K Filings

A RAG system that answers questions about disclosed risks across major 
US bank 10-K filings. Built as a portfolio project to demonstrate 
end-to-end RAG engineering: extraction, chunking, hybrid retrieval, 
grounded generation, and — most importantly — rigorous evaluation that 
surfaces real failure modes.

**Headline results** (Claude Haiku 4.5 generator, Ragas evaluation, n=16):
- Faithfulness: **0.68** — answers are mostly grounded but some claims 
  extend beyond retrieved context
- Answer relevancy: **0.71** — answers address the questions asked
- Context precision: **0.29** — retrieved chunks frequently contain 
  topically related but not directly useful content
- Context recall: **0.19** — top-5 retrieval frequently misses 
  information needed to fully answer

**What makes this project portfolio-worthy is not the score — it's the 
diagnostic methodology that produced it.** A custom evaluator rated this 
same system 5.0/5 on faithfulness. Ragas's claim-by-claim verification 
gave 0.68. The gap between those numbers is the project's most important 
finding.

---

## What This Demonstrates

- **Hybrid retrieval** — dense (ChromaDB + BGE embeddings) combined with 
  sparse (BM25) via Reciprocal Rank Fusion
- **Grounded generation** — structured prompt that enforces citation, 
  refusal on out-of-scope, and explicit answer-shape requirements for 
  comparison questions
- **Multi-tier evaluation** — first with a custom holistic evaluator, 
  then with Ragas's atomic-claim evaluator. The disagreement between 
  them is itself a finding.
- **Cross-model judging** — Claude Haiku 4.5 evaluates a Claude-generated 
  system. Reduces same-model bias compared to typical self-evaluation 
  setups.
- **Engineering rigor** — inspection JSONs at every pipeline stage, 
  versioned prompts (v1_baseline, v2_enforced_shape), cached pipeline 
  outputs so evaluation can be re-run cheaply.

---

## Architecture
User Query

│

▼

┌────────────────────┐

│   RAGPipeline      │  ← Orchestrator

└─────┬──────────────┘

│

├──► HybridRetriever

│      ├─ Dense:  ChromaDB + BAAI/bge-small-en-v1.5

│      ├─ Sparse: BM25 over tokenized chunks

│      └─ Fusion: Reciprocal Rank Fusion (k_constant=60)

│

▼

┌────────────────────┐

│  Generator         │  ← Claude Haiku 4.5

│                    │   (multi-provider: Anthropic, Gemini, DeepSeek)

└─────┬──────────────┘

│

▼

Answer + citations + diagnostics


A separate **Evaluator** layer runs two independent evaluation regimes:
- **Custom**: precision@k, recall@k, MRR for retrieval; faithfulness 
  and relevance via LLM-as-judge (holistic 1-5 scoring)
- **Ragas**: faithfulness, answer_relevancy, context_precision, 
  context_recall via Ragas framework with Claude Haiku as judge

---

## Data

**Source:** SEC EDGAR, accessed via the `sec-edgar-downloader` Python library.

**Corpus:** 9 fiscal year 2025 10-K filings, Item 1A (Risk Factors) only:
American Express (AXP), Bank of America (BAC), BlackRock (BLK), 
Capital One (COF), Goldman Sachs (GS), JPMorgan (JPM), Mastercard (MA), 
PayPal (PYPL), Visa (V). Total: 1,401 chunks.

Wells Fargo (WFC) extraction failed and is documented under Limitations.

**Format choice:** HTML from the EDGAR full-submission file, parsed with 
BeautifulSoup, rather than PDF — after discovering PDF extraction 
artifacts (letter-spacing tricks used by filing designers) that broke 
pypdf.

**Scope choice:** Item 1A only. Item 7 (MD&A) and 7A (market risk) 
excluded deliberately to keep the corpus focused. The downside of this 
choice surfaces in evaluation results — see "Why Context Metrics Are 
Low" below.

---

## Pipeline Stages

### 1. Extraction
- Downloads 10-Ks from SEC EDGAR by ticker
- Parses HTML to isolate Item 1A
- Produces per-filing extraction reports capturing character counts, 
  empty page counts, and non-ASCII ratios to detect extraction artifacts

### 2. Chunking
- Recursive character splitting (LangChain `RecursiveCharacterTextSplitter`)
- Chunk size 1200, overlap 150 — selected by retrieval quality on a 
  held-out question set
- Per-ticker chunk counts vary widely: COF has 280 chunks vs MA/V at 
  ~100 each. This imbalance turned out to matter for evaluation (see 
  context_precision discussion).

### 3. Embedding & Ingestion
- BAAI/bge-small-en-v1.5 embeddings (384 dimensions)
- Persisted to ChromaDB with metadata: ticker, accession number, 
  chunk index
- Metadata enables filtered retrieval (e.g., `where={"ticker": "JPM"}`)

### 4. Hybrid Retrieval
- Dense: ChromaDB top-25 candidate pool by cosine similarity
- Sparse: BM25 top-25 candidate pool by keyword scoring
- Fusion: RRF with k_constant=60, returning top-5 to the generator

### 5. Generation
- Default: Claude Haiku 4.5 (also supports Gemini Flash-Lite and DeepSeek)
- Two versioned prompt templates:
  - `v1_baseline`: original grounded-answer prompt
  - `v2_enforced_shape`: adds explicit structural rules for comparison 
    questions ("write a section per company, end with Key Differences")
- Programmatic post-processing extracts cited chunk numbers and flags 
  citation hallucinations

### 6. Evaluation
- 18-question golden test set authored against verified corpus content. 
  Distributed across single_company (6), comparison (5), specific_term 
  (3), paraphrase (2), out_of_scope (2).
- Custom evaluation: precision@5, recall@5, MRR, LLM-as-judge 
  faithfulness/relevance (1-5)
- Ragas evaluation: faithfulness, answer_relevancy, context_precision, 
  context_recall (continuous 0-1 scores)

---

## Evaluation: From Custom Metrics to Ragas

### Run 1 — Custom Evaluator (Gemini judge, n=4)

| Metric | Value |
|--------|-------|
| Avg precision@5 | 0.45 |
| Avg recall@5 | 0.75 |
| Avg faithfulness (1-5) | 5.0 |
| Avg relevance (1-5) | 2.33 |
| Citation hallucinations | 0 / 4 |
| Refusal accuracy | 4 / 4 |

**Initial reading:** Faithfulness looks perfect. Relevance is mixed. 
Job done?

**No.** Two issues with this conclusion:
1. n=4 is too small to draw conclusions
2. Holistic 1-5 scoring of faithfulness is too lenient — a single 
   judgment per answer can't catch claim-by-claim drift

### Run 2 — Ragas Evaluator (Claude Haiku 4.5 judge, n=16)

| Metric | Value | Healthy Range |
|--------|-------|---------------|
| Faithfulness | 0.68 | 0.75-0.90 |
| Answer relevancy | 0.71 | 0.70-0.85 |
| Context precision | 0.29 | 0.60-0.85 |
| Context recall | 0.19 | 0.60-0.85 |

**The gap from Run 1 to Run 2 is the project's most important finding.**

The custom evaluator gave faithfulness 5.0/5 (perfect). Ragas — using 
the same answers but a different evaluation methodology — gave 0.68. 
The custom evaluator had no analog at all for context_precision or 
context_recall.

Three real lessons from this:

**1. Holistic 1-5 scoring hides claim-level drift.**  
The custom judge read each answer and gave one number. Ragas breaks 
the answer into atomic claims and verifies each against the context. 
A 0.68 means roughly 32% of claims have some grounding issue — slight 
extrapolation, paraphrase that adds information, or interpretation 
beyond what the chunks say. The holistic judge couldn't see this 
because each individual claim looked plausible in context.

**2. Custom metrics had blind spots.**  
My custom precision@5 measured "did retrieved chunks come from the 
expected tickers?" That's a coarse signal. Ragas's context_precision 
asks the sharper question: "for this specific answer, was this specific 
chunk actually useful?" The drop from 0.45 to 0.29 reflects the gap 
between "topically related" and "actually useful."

**3. Cross-model judging changes the picture.**  
The custom run used Gemini-judging-Gemini. The Ragas run used 
Claude-judging-Claude. Both have same-family bias, but the migration 
to Claude was driven by Gemini free-tier instability — and produced 
the side benefit of more rigorous evaluation methodology.

---

## Why Context Metrics Are Low — Analysis

Context precision (0.29) and context recall (0.19) are both meaningfully 
below typical RAG system performance. Two hypotheses, each with 
different implications:

### Hypothesis A: Retrieval Quality

The retriever is bringing back chunks that are topically related to 
the question but don't actually contain the specific information needed 
to answer it. Supporting evidence:
- COF has 2x the chunk count of the median ticker, which may bias 
  retrieval toward COF in cross-company questions
- top_k=5 may be too small for questions where the relevant information 
  is scattered across multiple chunks
- RRF with k_constant=60 was not tuned — different values might 
  meaningfully change which chunks surface

If Hypothesis A dominates, fixes would be: re-ranking layer (cross-encoder), 
larger top_k, tuning RRF, or per-ticker retrieval balancing.

### Hypothesis B: Reference Answer Over-Specification

The Ragas context_recall metric compares retrieved chunks against the 
ground-truth answer. If the ground-truth answer contains claims that 
**aren't actually in the Item 1A corpus** (because they belong in 
Item 7 MD&A or financial footnotes), context_recall will be low even 
with perfect retrieval — because the necessary information isn't 
indexed at all.

This isn't theoretical — earlier development surfaced exactly this 
issue with a test question about LIBOR/SOFR transitions, which are 
typically discussed in Item 7, not Item 1A. The test question was 
revised. Other reference answers may have similar issues that weren't 
caught.

If Hypothesis B dominates, fixes would be: rewriting reference answers 
to only claim what's actually in Item 1A, or expanding the corpus to 
include Item 7.

### Likely Truth: Both, In Unknown Proportion

Diagnostic work would resolve this — inspect the 2-3 questions with 
the lowest context_recall and compare retrieved chunks to reference 
answers side by side. **This diagnostic is deferred** as a known 
limitation rather than implemented as a fix.

The honest framing: this is a system where evaluation has surfaced real 
issues, and the next iteration would address them with the diagnostic 
above. That iteration was paused due to environment instability (see 
Engineering Friction below).

---

## Engineering Friction Encountered

A real observation from this project: the LLM tooling ecosystem in 
mid-2026 is unstable. The following issues were encountered and 
documented during development:

- **Gemini free tier**: hit 10 RPM rate limits during eval; later hit 
  503 UNAVAILABLE during multi-question batch runs
- **DeepSeek**: removed free tier mid-project; migrated to Claude
- **Ragas + LangChain compatibility**: Ragas imports 
  `langchain_community.chat_models.vertexai` unconditionally, which 
  was deprecated/removed in recent langchain-community versions. 
  Resolved by pinning langchain-community to <0.3.0.
- **Windows file locks during installs**: pip/uv installs failed when 
  the venv was being used by an active Jupyter kernel; broke numpy/
  pandas binary compatibility
- **Broken venv**: pyvenv.cfg disappeared during a chain of installs; 
  required full venv rebuild

These are mentioned not as complaints but as honest documentation. 
Production AI engineering involves significant time on dependency 
management — knowing when to switch tools versus persist with debugging 
is itself a judgment skill.

---

## How to Run

### Prerequisites
- Python 3.11
- `uv` for dependency management
- Anthropic API key — generation and the Ragas judge
- Gemini API key — the custom evaluator's judge, and required by the 
  shared notebook setup (`notebooks/_setup.py`)
- DeepSeek API key — optional, only if using the DeepSeek generator

### Setup
```bash
git clone <repo-url>
cd financial_risk_rag
uv sync
cp .env.example .env
# Edit .env to add ANTHROPIC_API_KEY and GEMINI_API_KEY
```

### Run pipeline (smoke test)
```bash
uv run jupyter notebook notebooks/04_test_pipeline.ipynb
```

### Reproduce custom evaluation
```bash
uv run jupyter notebook notebooks/05_run_evaluation.ipynb
```

### Reproduce Ragas evaluation
```bash
uv run jupyter notebook notebooks/ragas_baseline.ipynb
```

---

## Repository Structure

```
financial_risk_rag/
├── src/
│   ├── extraction.py           # Filing discovery + Item 1A extraction from HTML
│   ├── chunking.py             # Recursive character chunking with metadata
│   ├── ingestion.py            # BGE embeddings → ChromaDB
│   ├── retrieval.py            # HybridRetriever (dense + BM25 + RRF)
│   ├── generation.py           # Multi-provider Generator (Claude/Gemini/DeepSeek)
│   ├── pipeline.py             # End-to-end orchestrator
│   └── evaluation.py           # Custom metrics + LLM-as-judge
├── data/
│   ├── raw_pdfs/sec-edgar-filings/     # Raw EDGAR submissions per ticker (gitignored)
│   ├── extracted_text/                 # Full 10-K + Item 1A text per filing,
│   │                                     plus extraction_report.json (gitignored)
│   ├── chunks/chunking.json            # Chunking inspection report
│   ├── ingestion/ingestion_report.json # Ingestion inspection report
│   ├── test_set/test_set.json          # 18-question golden test set
│   ├── eval_run_baseline.json          # Baseline custom evaluation results
│   └── pipeline_outputs_claude_v2.pkl  # Cached pipeline runs (so evaluation can be
│                                         re-run without re-calling Claude)
├── notebooks/
│   ├── _setup.py                       # Shared notebook setup (paths, .env, retriever)
│   ├── 00_SEC_download.ipynb           # Downloads 10-K filings from SEC EDGAR
│   ├── 01_extraction_exploration.ipynb # Extraction exploration
│   ├── 02_chunking_tests.ipynb         # Chunk size / overlap experiments
│   ├── 03_ingestion_check.ipynb        # Sanity checks on the ChromaDB collection
│   ├── 04_retrieval_experiments.ipynb  # Retrieval quality experiments
│   ├── 05_test_pipeline.ipynb          # Pipeline smoke test
│   ├── 06_run_evaluation.ipynb         # Custom evaluation
│   ├── ABtest_evaluation.ipynb         # v1_baseline vs v2_enforced_shape prompt A/B
│   ├── corpus_audit.ipynb              # Per-ticker corpus coverage audit
│   └── ragas_baseline.ipynb            # Ragas evaluation
├── chroma_db/                          # Persistent vector store (gitignored)
├── .env.example                        # Template for required API keys
├── pyproject.toml                      # Dependencies (managed with uv)
├── uv.lock
└── CLAUDE.md                           # Context for AI coding tools
```

---

## Engineering Lessons

**Multiple evaluation regimes are not redundant.** Custom holistic 
scoring and Ragas atomic-claim scoring measure related-but-distinct 
properties. Both are useful. Running only one would have hidden either 
the high-level "the system looks fine" signal or the granular "claim 
N has issues" signal. The disagreement between them is where the 
actionable diagnostics live.

**Reference answer quality is half of evaluation.** A test set where 
reference answers ask for information not in the indexed corpus produces 
low recall scores that look like retrieval failures but are actually 
test set failures. The LIBOR/SOFR question was the cleanest example. 
Some context_recall = 0.19 is likely the same problem, undiagnosed.

**Cross-model judging > self-judging, but isn't a panacea.** Migrating 
the judge from Gemini to Claude (different model family) reduces 
correlation bias compared to same-family judging. But both are still 
LLMs, and both can be wrong in correlated ways. Human-graded 
calibration on a sample would close the loop. Not yet implemented.

**Versioning is cheap insurance.** The two prompt versions 
(`v1_baseline`, `v2_enforced_shape`) and the cached pipeline outputs 
made it possible to re-run evaluation against different judges without 
re-spending API budget on generation. This pattern paid for itself 
multiple times.

---

## Limitations

**Scope:**
- Item 1A only. Item 7 (MD&A) and 7A (market risk) excluded. Some test 
  questions may require content from those sections, contributing to 
  low context_recall.
- Single fiscal year (2025). Multi-year would enable trend questions.
- Wells Fargo extraction failed. Root cause not diagnosed.

**Evaluation:**
- n=18 is modest; larger would tighten confidence intervals on Ragas 
  scores.
- LLM-as-judge calibration against human-graded samples not yet 
  performed.
- A/B comparison between v1_baseline and v2_enforced_shape prompts not 
  yet run.
- Context precision/recall diagnostic (Hypothesis A vs B above) not 
  yet performed.

**Production-readiness gaps:**
- No re-ranking layer (cross-encoder would likely improve 
  context_precision)
- No latency optimization (current end-to-end ~2-3s, dominated by 
  generation)
- No streaming output
- No UI — currently notebook-only

**Status:** Development on this project is paused. The system is 
functional and the evaluation findings are real. Future iterations 
would address the diagnostic work, prompt A/B, and the production-
readiness gaps.

---

## Built With

Python 3.11, ChromaDB, sentence-transformers (BAAI/bge-small-en-v1.5), 
rank-bm25, LangChain text splitters, Claude Haiku 4.5 (via Anthropic 
API), Ragas, sec-edgar-downloader, BeautifulSoup.

Developed with assistance from Claude Code (configuration in `CLAUDE.md`).