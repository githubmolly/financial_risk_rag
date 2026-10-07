import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

_TICKER_TO_COMPANY: dict[str, str] = {
    "AXP": "American Express",
    "BAC": "Bank of America",
    "BLK": "BlackRock",
    "COF": "Capital One",
    "GS": "Goldman Sachs",
    "JPM": "JPMorgan Chase",
    "MA": "Mastercard",
    "PYPL": "PayPal",
    "V": "Visa",
    "WFC": "Wells Fargo",
}

# Accession format: {CIK}-{YY}-{seq}  →  first "-NN-" group is the filing year.
_ACCESSION_YEAR_RE = re.compile(r"-(\d{2})-")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str       # "{ticker}_{accession}_{file_type}_{index}"
    text: str
    source: str         # original filename
    ticker: str
    company: str
    filing_year: int    # e.g. 2026
    file_type: str      # "risk_factors"
    item: str           # "1A" for all risk_factors files
    chunk_index: int    # 0-based within this file
    char_start: int     # byte-exact position in the source file
    char_end: int

    def to_chroma_metadata(self) -> dict:
        """Flat dict of primitive types suitable for ChromaDB metadata."""
        return {
            "source": self.source,
            "ticker": self.ticker,
            "company": self.company,
            "filing_year": self.filing_year,
            "file_type": self.file_type,
            "item": self.item,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_filename(path: Path) -> dict:
    """
    Extract structured metadata from a risk_factors filename.
    Expected stem: "{TICKER}_{accession}_{file_type}"
    e.g.  "JPM_0001628280-26-008131_risk_factors"
    """
    stem = path.stem
    parts = stem.split("_", 2)
    ticker = parts[0]
    accession = parts[1] if len(parts) > 1 else ""
    file_type = parts[2] if len(parts) > 2 else "unknown"

    m = _ACCESSION_YEAR_RE.search(accession)
    filing_year = 2000 + int(m.group(1)) if m else 0

    return {
        "ticker": ticker,
        "accession": accession,
        "file_type": file_type,
        "filing_year": filing_year,
        "company": _TICKER_TO_COMPANY.get(ticker, ticker),
    }


# ---------------------------------------------------------------------------
# Offset-aware splitting
# ---------------------------------------------------------------------------

def _split_with_offsets(
    text: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[tuple[str, int, int]]:
    """
    Run the splitter and return (chunk_text, char_start, char_end) for each chunk.

    RecursiveCharacterTextSplitter strips whitespace from chunks, so we anchor
    each chunk's position by searching for its first 40 characters sequentially
    through the original text.  The cursor advances to just before the expected
    start of the next chunk (accounting for overlap) to avoid false matches on
    repeated phrases.
    """
    raw_chunks = splitter.split_text(text)
    result: list[tuple[str, int, int]] = []
    search_from = 0

    for chunk in raw_chunks:
        anchor = chunk[:40]
        idx = text.find(anchor, search_from)
        if idx == -1:
            # Back up a full chunk window and retry — handles rare edge cases
            # where a splitter boundary sits inside the safety margin.
            idx = text.find(anchor, max(0, search_from - CHUNK_SIZE))
        start = idx if idx != -1 else search_from
        end = start + len(chunk)
        result.append((chunk, start, end))
        # Next search starts just before where the next chunk begins.
        # Subtracting (CHUNK_OVERLAP + 20) from end puts the cursor at the
        # approximate overlap start with a small safety margin.
        search_from = max(0, end - CHUNK_OVERLAP - 20)

    return result


# ---------------------------------------------------------------------------
# Core chunking function
# ---------------------------------------------------------------------------

def chunk_document(text: str, file_meta: dict) -> list[Chunk]:
    """Chunk one document and return fully-annotated Chunk objects."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    pieces = _split_with_offsets(text, splitter)
    source = (
        f"{file_meta['ticker']}_{file_meta['accession']}"
        f"_{file_meta['file_type']}.txt"
    )

    chunks: list[Chunk] = []
    for idx, (chunk_text, char_start, char_end) in enumerate(pieces):
        chunks.append(Chunk(
            chunk_id=f"{file_meta['ticker']}_{file_meta['accession']}"
                     f"_{file_meta['file_type']}_{idx}",
            text=chunk_text,
            source=source,
            ticker=file_meta["ticker"],
            company=file_meta["company"],
            filing_year=file_meta["filing_year"],
            file_type=file_meta["file_type"],
            item="1A",
            chunk_index=idx,
            char_start=char_start,
            char_end=char_end,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_chunking(
    extracted_dir: str | Path = "data/extracted_text",
    chunks_dir: str | Path = "data/chunks",
) -> list[Chunk]:
    extracted_dir = Path(extracted_dir)
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    risk_files = sorted(extracted_dir.glob("*_risk_factors.txt"))
    print(f"Found {len(risk_files)} risk_factors file(s)")

    all_chunks: list[Chunk] = []
    for path in risk_files:
        file_meta = _parse_filename(path)
        text = path.read_text(encoding="utf-8")
        chunks = chunk_document(text, file_meta)
        all_chunks.extend(chunks)
        print(f"  {file_meta['ticker']:6}  {len(chunks):3} chunks"
              f"  ({file_meta['company']}, {file_meta['filing_year']})")

    out_path = chunks_dir / "chunks.json"
    out_path.write_text(
        json.dumps([asdict(c) for c in all_chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nTotal: {len(all_chunks)} chunks  ->  {out_path}")
    return all_chunks


if __name__ == "__main__":
    import sys
    _extracted = sys.argv[1] if len(sys.argv) > 1 else "data/extracted_text"
    _chunks = sys.argv[2] if len(sys.argv) > 2 else "data/chunks"
    run_chunking(_extracted, _chunks)
