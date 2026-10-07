import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Stage 1 – Discovery
# ---------------------------------------------------------------------------

def discover_filings(data_dir: Path) -> list[dict]:
    edgar_root = data_dir / "raw_pdfs" / "sec-edgar-filings"
    filings = []
    for ticker_dir in sorted(edgar_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        form_dir = ticker_dir / "10-K"
        if not form_dir.exists():
            continue
        for accession_dir in sorted(form_dir.iterdir()):
            submission = accession_dir / "full-submission.txt"
            if submission.exists():
                filings.append({
                    "ticker": ticker_dir.name,
                    "accession": accession_dir.name,
                    "path": submission,
                })
    return filings


# ---------------------------------------------------------------------------
# Stage 2 – SGML unwrap
# ---------------------------------------------------------------------------

_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)


def extract_primary_document(sgml: str) -> str | None:
    """Return the HTML body of the first TYPE=10-K SEQUENCE=1 document."""
    for doc_m in _DOC_RE.finditer(sgml):
        body = doc_m.group(1)
        type_m = re.search(r"<TYPE>(.+)", body)
        seq_m = re.search(r"<SEQUENCE>(.+)", body)
        if not (type_m and seq_m):
            continue
        if type_m.group(1).strip() == "10-K" and seq_m.group(1).strip() == "1":
            text_m = _TEXT_RE.search(body)
            if text_m:
                return text_m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Stage 3 – HTML → plain text
# ---------------------------------------------------------------------------

def parse_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "meta", "link", "head"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Stage 4 – Letter-spacing fix
# ---------------------------------------------------------------------------

_LS_WORD_RE = re.compile(r"(?<!\w)([A-Za-z](?: [A-Za-z]){2,})(?!\w)")
_COMMON_SINGLES = frozenset({"a", "i", "I"})


def _has_letter_spacing(text: str, threshold: float = 0.20) -> bool:
    tokens = text.split()
    if len(tokens) < 100:
        return False
    single_alpha = sum(
        1 for t in tokens
        if len(t) == 1 and t.isalpha() and t not in _COMMON_SINGLES
    )
    return (single_alpha / len(tokens)) > threshold


def _fix_letter_spacing(text: str) -> tuple[str, int]:
    count = 0

    def _collapse(m: re.Match) -> str:
        nonlocal count
        count += 1
        return m.group(0).replace(" ", "")

    return _LS_WORD_RE.sub(_collapse, text), count


# ---------------------------------------------------------------------------
# Stage 5 – Artifact cleaning
# ---------------------------------------------------------------------------

# Matches "Table [of Contents]" page-nav stubs injected by the HTML.
# Handles all split layouts found across EDGAR filings:
#   • "\nTable\n"               standalone word
#   • "\nTable of Contents\n"   single line
#   • "\nTable\nof Contents\n"  split after "Table"
#   • "\nTable of\nContents\n"  split after "of"
# Page numbers (1-3 digits) are stripped before this runs, so they appear
# as blank lines and are consumed by the preceding-blank-lines group.
_NAV_STUB_RE = re.compile(
    r"\n"
    r"(?:[ \t]*\n)*"                                    # optional preceding blank lines
    r"[ \t]*Table(?:[ \t]+of(?:[ \t]+Contents)?)?[ \t]*\n"  # "Table [of [Contents]]"
    r"(?:[ \t]*(?:of[ \t]+)?Contents[ \t]*\n)?"         # optional continuation line
    r"(?:[ \t]*\n)*",                                   # optional trailing blank lines
    re.IGNORECASE,
)


def _nav_replace(m: re.Match) -> str:
    # Look at the last non-blank line before this stub to decide whether
    # it ended a sentence.  If yes, keep a paragraph break; if not,
    # the stub was interrupting a sentence — reconnect with a single newline.
    preceding = m.string[: m.start()].rstrip("\n")
    last_line = (preceding.rsplit("\n", 1)[-1] if "\n" in preceding else preceding).strip()
    return "\n\n" if (last_line and last_line[-1] in ".!?;:") else "\n"


def clean_artifacts(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    # Page numbers before nav stubs: removed digits become blank lines,
    # which the nav stub regex then consumes as part of its match.
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = _NAV_STUB_RE.sub(_nav_replace, text)

    lines = text.splitlines()
    freq = Counter(ln.strip() for ln in lines if len(ln.strip()) > 10)
    repeated = {ln for ln, n in freq.items() if n >= 5}
    lines = [ln for ln in lines if ln.strip() not in repeated]
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 6 – Item 1A extraction
# ---------------------------------------------------------------------------

_ITEM_1A_RE = re.compile(
    r"item\s+1a\.?\s*risk\s+factors",
    re.IGNORECASE,
)
_ITEM_END_RE = re.compile(
    r"(?:^|\n)\s*item\s+(?:1b|2)\.?\s*",
    re.IGNORECASE,
)

_MIN_SECTION_CHARS = 2000


def extract_item_1a(text: str) -> str | None:
    """Return the full Item 1A text, or None if not found / too short."""
    starts = list(_ITEM_1A_RE.finditer(text))
    if not starts:
        return None

    ends = list(_ITEM_END_RE.finditer(text))

    for start_m in starts:
        content_start = start_m.end()
        next_end = next((m for m in ends if m.start() > content_start), None)
        section_len = (next_end.start() if next_end else len(text)) - content_start
        if section_len >= _MIN_SECTION_CHARS:
            end_pos = next_end.start() if next_end else len(text)
            return text[start_m.start():end_pos].strip()

    return None


# ---------------------------------------------------------------------------
# Core per-filing processor
# ---------------------------------------------------------------------------

def _process_filing(filing: dict, output_dir: Path) -> dict:
    ticker: str = filing["ticker"]
    accession: str = filing["accession"]
    path: Path = filing["path"]
    issues: list[str] = []
    stem = f"{ticker}_{accession}"

    try:
        sgml = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {
            "ticker": ticker, "accession": accession,
            "status": "error", "issues": [f"read_error: {exc}"],
        }

    html = extract_primary_document(sgml)
    if html is None:
        return {
            "ticker": ticker, "accession": accession,
            "status": "error", "issues": ["sgml_parse_failed"],
        }

    text = parse_html_to_text(html)

    ls_fixes = 0
    if _has_letter_spacing(text):
        text, ls_fixes = _fix_letter_spacing(text)
        issues.append(f"letter_spacing_detected: {ls_fixes} fixes applied")

    text = clean_artifacts(text)

    item_1a = extract_item_1a(text)
    if item_1a is None:
        issues.append("item_1a_missing: excluded from output")
        return {
            "ticker": ticker,
            "accession": accession,
            "status": "excluded",
            "char_count": len(text),
            "word_count": len(text.split()),
            "item_1a_char_count": 0,
            "letter_spacing_fixes": ls_fixes,
            "issues": issues,
        }

    (output_dir / f"{stem}_10K.txt").write_text(text, encoding="utf-8")
    (output_dir / f"{stem}_risk_factors.txt").write_text(item_1a, encoding="utf-8")

    return {
        "ticker": ticker,
        "accession": accession,
        "status": "ok",
        "char_count": len(text),
        "word_count": len(text.split()),
        "item_1a_char_count": len(item_1a),
        "letter_spacing_fixes": ls_fixes,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_extraction(
    data_dir: str | Path = "data",
    output_dir: str | Path = "data/extracted_text",
) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filings = discover_filings(data_dir)
    print(f"Found {len(filings)} filing(s)")

    report = []
    for filing in filings:
        print(f"  Processing {filing['ticker']} ({filing['accession']})...", end=" ", flush=True)
        result = _process_filing(filing, output_dir)
        status = result["status"]
        warnings = result.get("issues", [])
        print(status + (f"  WARN: {warnings}" if warnings else ""))
        report.append(result)

    report_path = output_dir / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ok = sum(1 for r in report if r["status"] == "ok")
    excluded = sum(1 for r in report if r["status"] == "excluded")
    errors = sum(1 for r in report if r["status"] == "error")
    print(f"\nDone — ok:{ok}  excluded:{excluded}  errors:{errors}")
    print(f"Report: {report_path}")

    return {
        "total": len(report),
        "ok": ok,
        "excluded": excluded,
        "errors": errors,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    import sys

    _data = sys.argv[1] if len(sys.argv) > 1 else "data"
    _out = sys.argv[2] if len(sys.argv) > 2 else "data/extracted_text"
    run_extraction(_data, _out)
