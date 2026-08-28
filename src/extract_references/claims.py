"""
claims.py
Deterministic (regex-based, no LLM) detection and resolution of in-text
citation markers against the parsed reference list, plus the text
segmentation (paragraphs / sentences) used by `claim_extraction.py`.

Citation markers are treated as independent objects, never assumed to equal
a reference's position in the list:
  - Numbered style ('[3]', '[1, 2]', '[4-6]') is resolved by parsing the
    *original* in-text number out of the reference's own `raw_text` prefix
    (e.g. "[3] Smith et al. ..."), so resolution survives re-indexing after
    implausible entries are filtered out upstream.
  - Author-year style ('(Wei et al., 2022)', 'Wei et al. (2022)', '(Wei,
    Wang, and Schuurmans, 2022)', '(Wei et al., 2022a, 2022b)', '(e.g., Wei
    et al., 2022; Snell et al., 2024)') is resolved by (first-author
    surname, year) against the parsed citation list, with year-suffix
    ('2022a' / '2022b') used to disambiguate when two references share the
    same first author and year.

Nothing here ever guesses silently: a marker that cannot be resolved is
left untouched in the text (status='unresolved'), and one that matches more
than one reference without a usable suffix is reported as 'ambiguous'
rather than picking arbitrarily.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schemas import ExtractedCitation

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ParsedCitationRef:
    """One individual citation parsed out of a marker (a marker may hold several)."""
    number: Optional[int] = None          # numbered style
    author_key: Optional[str] = None      # author-year style: lowercased surname
    year: Optional[int] = None            # author-year style
    suffix: Optional[str] = None          # e.g. 'a' in "2022a"


@dataclass
class CitationSpan:
    raw: str
    start: int
    end: int
    style: str  # "numbered" | "author_year"
    refs: List[ParsedCitationRef] = field(default_factory=list)


@dataclass
class ResolvedRef:
    ref: ParsedCitationRef
    status: str  # "resolved" | "ambiguous" | "unresolved"
    resolved_ref_id: Optional[str] = None
    candidates: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_ABBREV_LOOKBEHINDS = [
    r"(?<!et al\.)",
    r"(?<!e\.g\.)",
    r"(?<!i\.e\.)",
    r"(?<! cf\.)",
    r"(?<! vs\.)",
    r"(?<! Fig\.)",
    r"(?<! fig\.)",
    r"(?<! Eq\.)",
    r"(?<! eq\.)",
    r"(?<! No\.)",
    r"(?<! Sec\.)",
    r"(?<! pp\.)",
    r"(?<! Vol\.)",
    r"(?<![A-Z]\.)",  # single-letter initials, e.g. "J. Smith"
]

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])" + "".join(_ABBREV_LOOKBEHINDS) + r'\s+(?=[A-Z0-9"\'(\[])'
)


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split `text` into (sentence, start_offset, end_offset) triples.

    Best-effort, regex-based. Offsets are into the original `text`.
    """
    if not text:
        return []

    sentences: List[Tuple[str, int, int]] = []
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        end = m.start()
        chunk = text[start:end].strip()
        if chunk:
            real_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            sentences.append((chunk, real_start, real_start + len(chunk)))
        start = m.end()

    tail = text[start:].strip()
    if tail:
        real_start = start + (len(text[start:]) - len(text[start:].lstrip()))
        sentences.append((tail, real_start, real_start + len(tail)))

    return sentences


def tag_sentences(text: str, start_id: int) -> Tuple[str, List[Tuple[str, str]], int]:
    """Prefix each sentence in `text` with a global '[S<n>]' tag.

    Returns (tagged_text, [(sentence_id, sentence_text), ...], next_start_id).
    """
    sentences = split_sentences(text)
    if not sentences:
        return text, [], start_id

    parts: List[str] = []
    id_map: List[Tuple[str, str]] = []
    last_end = 0
    sid = start_id
    for sent, s_start, s_end in sentences:
        parts.append(text[last_end:s_start])
        tag = f"S{sid}"
        parts.append(f"[{tag}] {sent}")
        id_map.append((tag, sent))
        last_end = s_end
        sid += 1
    parts.append(text[last_end:])
    return "".join(parts), id_map, sid


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def split_paragraphs(body_text: str, min_len: int = 40) -> List[str]:
    """Split markdown body text into prose paragraphs.

    Filters out headings, table rows, image references, and display-math
    blocks, plus fragments shorter than `min_len` chars (usually captions
    or stray labels). Internal whitespace/line-breaks are collapsed to
    single spaces so PDF-extraction line wraps don't fragment sentences.
    """
    if not body_text:
        return []

    raw_blocks = re.split(r"\n\s*\n+", body_text)
    paragraphs: List[str] = []
    for block in raw_blocks:
        collapsed = re.sub(r"\s+", " ", block).strip()
        if not collapsed:
            continue
        if collapsed.startswith(("#", "|", "![", "$$")):
            continue
        if len(collapsed) < min_len:
            continue
        paragraphs.append(collapsed)
    return paragraphs


# ---------------------------------------------------------------------------
# Numbered-citation markers: [1], [1,2], [1-3], [1, 2-4]
# ---------------------------------------------------------------------------

_NUMBERED_MARKER_RE = re.compile(
    r"\[(\d{1,3}(?:\s*[-–]\s*\d{1,3})?(?:\s*,\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?)*)\]"
)


def _expand_numbered_group(inner: str) -> List[int]:
    """Expand '1, 2, 4-6' -> [1, 2, 4, 5, 6]."""
    numbers: List[int] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
        if range_match:
            lo, hi = int(range_match.group(1)), int(range_match.group(2))
            if lo <= hi and (hi - lo) < 50:  # guard against pathological ranges
                numbers.extend(range(lo, hi + 1))
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def find_numbered_spans(body_text: str) -> List[CitationSpan]:
    """Find all '[N]' / '[N,M]' / '[N-M]' style markers as CitationSpans."""
    spans: List[CitationSpan] = []
    for m in _NUMBERED_MARKER_RE.finditer(body_text):
        numbers = _expand_numbered_group(m.group(1))
        if numbers:
            spans.append(
                CitationSpan(
                    raw=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    style="numbered",
                    refs=[ParsedCitationRef(number=n) for n in numbers],
                )
            )
    return spans


# ---------------------------------------------------------------------------
# Author-year citation markers
# ---------------------------------------------------------------------------

_SURNAME_RE = re.compile(r"[A-Z][A-Za-zÀ-ſ'\-]+")
_YEAR_TOKEN_RE = re.compile(r"\b(\d{4})([a-z])?\b")
_LEADING_FILLER_RE = re.compile(r"^(e\.g\.,?|i\.e\.,?|cf\.,?|see,?)\s*", re.IGNORECASE)

_SURNAME = r"[A-Z][A-Za-zÀ-ſ'\-]+"
# Author phrase used to anchor a *narrative* citation right before "(year)":
# "Wei", "Wei et al.", "Wei and Jones", "Wei & Jones".
_NARRATIVE_AUTHOR_PHRASE = (
    rf"{_SURNAME}(?:\s+(?:et\s*al\.?|and\s+{_SURNAME}|&\s*{_SURNAME}))?"
)
_NARRATIVE_RE = re.compile(rf"({_NARRATIVE_AUTHOR_PHRASE})\s*\(([^()]{{2,120}})\)")

# A whole parenthetical block that looks like a citation group, e.g.
# "(Smith et al., 2020; Jones, 2019)"
_PARENS_GROUP_RE = re.compile(r"\(([^()]{4,400}?\d{4}[a-z]?[^()]{0,300}?)\)")


def _leading_surname(segment: str) -> Optional[str]:
    """First capitalized surname-like token at the start of `segment`.

    Deliberately does NOT try to be clever about "and"/"&" connectors —
    for a full author list like "Wei, Wang, and Schuurmans, 2022" this
    correctly returns "Wei" (the true first author) rather than drifting to
    a later name, which a single greedy "surname (connector) year" regex
    tends to do.
    """
    s = _LEADING_FILLER_RE.sub("", segment.strip())
    m = _SURNAME_RE.match(s)
    return m.group(0) if m else None


def _all_years(segment: str) -> List[Tuple[int, Optional[str]]]:
    """All 4-digit year tokens (with optional suffix letter) found in `segment`."""
    return [(int(y), suf or None) for y, suf in _YEAR_TOKEN_RE.findall(segment)]


def find_author_year_spans(body_text: str) -> List[CitationSpan]:
    """Find author-year style markers, both parenthetical-group and narrative."""
    spans: List[CitationSpan] = []
    covered: List[Tuple[int, int]] = []

    for m in _PARENS_GROUP_RE.finditer(body_text):
        inner = m.group(1)
        refs: List[ParsedCitationRef] = []
        for segment in inner.split(";"):
            surname = _leading_surname(segment)
            years = _all_years(segment)
            if surname and years:
                key = surname.lower()
                for year, suffix in years:
                    refs.append(ParsedCitationRef(author_key=key, year=year, suffix=suffix))
        if refs:
            spans.append(
                CitationSpan(raw=m.group(0), start=m.start(), end=m.end(), style="author_year", refs=refs)
            )
            covered.append((m.start(), m.end()))

    for m in _NARRATIVE_RE.finditer(body_text):
        if any(c_start <= m.start() and m.end() <= c_end for c_start, c_end in covered):
            continue  # already captured as part of a parenthetical group
        surname = _leading_surname(m.group(1))
        years = _all_years(m.group(2))
        if not (surname and years):
            continue
        key = surname.lower()
        refs = [ParsedCitationRef(author_key=key, year=year, suffix=suffix) for year, suffix in years]
        spans.append(
            CitationSpan(raw=m.group(0), start=m.start(), end=m.end(), style="author_year", refs=refs)
        )

    spans.sort(key=lambda s: s.start)
    return spans


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------

def detect_citation_style(body_text: str) -> str:
    """Return 'numbered', 'author_year', 'mixed', or 'unknown'."""
    numbered_count = len(find_numbered_spans(body_text))
    author_year_count = len(find_author_year_spans(body_text))

    if numbered_count == 0 and author_year_count == 0:
        return "unknown"
    if numbered_count >= 3 and numbered_count > author_year_count * 2:
        return "numbered"
    if author_year_count >= 3 and author_year_count > numbered_count * 2:
        return "author_year"
    return "mixed"


# ---------------------------------------------------------------------------
# Resolution: marker -> canonical ref_id
# ---------------------------------------------------------------------------

_LEADING_NUMBER_RE = re.compile(r"^\s*[\[\(]?\s*(\d{1,3})\s*[\]\).]")
_SUFFIX_ORDER = "abcdefghij"


def build_number_index(citations: List[ExtractedCitation]) -> Dict[int, str]:
    """Map the original in-text number (parsed from '[N] ...' in raw_text) to ref_id.

    Citations whose raw_text has no parseable leading number are skipped.
    Because this parses the number straight out of the source text rather
    than relying on list position, it stays correct even after upstream
    filtering re-indexes the surviving citations to contiguous R1, R2, ...
    """
    mapping: Dict[int, str] = {}
    for citation in citations:
        if not citation.raw_text:
            continue
        m = _LEADING_NUMBER_RE.match(citation.raw_text)
        if m:
            mapping[int(m.group(1))] = citation.ref_id
    return mapping


def _first_author_surname(citation: ExtractedCitation) -> Optional[str]:
    if not citation.authors:
        return None
    first = citation.authors[0].strip()
    if not first:
        return None
    return first.split()[-1].lower()


def build_author_year_index(citations: List[ExtractedCitation]) -> Dict[Tuple[str, int], List[str]]:
    """Map (first_author_surname_lower, year) -> [ref_id, ...] (usually length 1).

    A key with more than one ref_id means two references share the same
    first author and year — those are only resolved when the in-text
    marker carries a disambiguating suffix ('a'/'b'); see `resolve_ref`.
    Order follows the order `citations` was given in, which for an
    author-year bibliography is normally already alphabetical/chronological,
    matching how '2022a' vs '2022b' would have been assigned.
    """
    index: Dict[Tuple[str, int], List[str]] = {}
    for citation in citations:
        if citation.year is None:
            continue
        surname = _first_author_surname(citation)
        if not surname:
            continue
        index.setdefault((surname, citation.year), []).append(citation.ref_id)
    return index


def resolve_ref(
    ref: ParsedCitationRef,
    number_index: Dict[int, str],
    author_year_index: Dict[Tuple[str, int], List[str]],
) -> ResolvedRef:
    if ref.number is not None:
        ref_id = number_index.get(ref.number)
        if ref_id:
            return ResolvedRef(ref=ref, status="resolved", resolved_ref_id=ref_id)
        return ResolvedRef(ref=ref, status="unresolved")

    if ref.author_key is not None and ref.year is not None:
        candidates = author_year_index.get((ref.author_key, ref.year), [])
        if not candidates:
            return ResolvedRef(ref=ref, status="unresolved")
        if len(candidates) == 1:
            return ResolvedRef(ref=ref, status="resolved", resolved_ref_id=candidates[0])
        if ref.suffix and ref.suffix in _SUFFIX_ORDER:
            pos = _SUFFIX_ORDER.index(ref.suffix)
            if pos < len(candidates):
                return ResolvedRef(ref=ref, status="resolved", resolved_ref_id=candidates[pos])
        return ResolvedRef(ref=ref, status="ambiguous", candidates=list(candidates))

    return ResolvedRef(ref=ref, status="unresolved")


# ---------------------------------------------------------------------------
# Canonical tagging: replace resolved markers with '<CIT:ref_id>' inline
# ---------------------------------------------------------------------------

def tag_citations(
    text: str,
    citations: List[ExtractedCitation],
    style: Optional[str] = None,
) -> Tuple[str, List[ResolvedRef]]:
    """Replace every fully-resolved citation marker in `text` with one or
    more inline '<CIT:ref_id>' tags (concatenated when a marker holds
    several works, e.g. "(Wei et al., 2022; Snell et al., 2024)").

    A marker that is only partially resolved, ambiguous, or unresolved is
    left as its original raw text — downstream consumers (the claim LLM)
    still see it, just not canonicalized.

    Returns (tagged_text, all_resolved_refs) so callers can log/inspect
    ambiguous or unresolved markers.
    """
    if not text or not citations:
        return text, []

    resolved_style = style or detect_citation_style(text)
    spans: List[CitationSpan] = []
    if resolved_style in ("numbered", "mixed"):
        spans.extend(find_numbered_spans(text))
    if resolved_style in ("author_year", "mixed"):
        spans.extend(find_author_year_spans(text))
    spans.sort(key=lambda s: s.start)

    number_index = build_number_index(citations) if resolved_style in ("numbered", "mixed") else {}
    author_year_index = (
        build_author_year_index(citations) if resolved_style in ("author_year", "mixed") else {}
    )

    out: List[str] = []
    last = 0
    all_resolved: List[ResolvedRef] = []
    for span in spans:
        if span.start < last:
            continue  # overlaps a span already consumed
        resolved_refs = [resolve_ref(r, number_index, author_year_index) for r in span.refs]
        all_resolved.extend(resolved_refs)

        out.append(text[last:span.start])
        if resolved_refs and all(r.status == "resolved" for r in resolved_refs):
            out.append("".join(f"<CIT:{r.resolved_ref_id}>" for r in resolved_refs))
        else:
            out.append(text[span.start:span.end])
        last = span.end
    out.append(text[last:])

    return "".join(out), all_resolved
