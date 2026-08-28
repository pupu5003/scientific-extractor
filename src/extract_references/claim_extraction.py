"""
claim_extraction.py
LLM-driven, paragraph-level claim extraction with citation alignment.

Why paragraph-level, not sentence-level: a citation frequently anchors on
one sentence ("Chain-of-thoughts (CoT) <CIT:R17>") while a *different*
sentence nearby restates or elaborates the same idea using a pronoun or
short-form reference ("CoTs seek to emulate human reasoning by ...") without
repeating the citation tag. Sentence-by-sentence regex matching cannot see
that link; an LLM reading the whole paragraph can.

Pipeline:
  1. Resolve in-text citation markers to canonical reference IDs
     (deterministic, see `claims.py`) and rewrite them as inline
     '<CIT:ref_id>' tags — the LLM never has to understand APA/IEEE/etc.
  2. Split the body into paragraphs, tag each sentence '[S1] ... [S2] ...'.
  3. Ask the LLM to extract atomic claims per paragraph, distinguishing
     'explicit' citations (tagged in the claim's own sentence) from
     'inherited' ones (tagged only in an earlier sentence of the same
     paragraph, linked by discourse).
  4. Assemble global `Claim` objects, filtering out any ref id the LLM
     didn't actually see tagged in that paragraph (anti-hallucination) and
     any sentence id it didn't see (anti-hallucination for source ids too).
"""
from __future__ import annotations

import asyncio
from typing import List, Protocol, Tuple

from .claims import detect_citation_style, split_paragraphs, tag_citations, tag_sentences
from .schemas import Claim, ExtractedCitation, ParagraphClaimCollection


class ClaimLLMClient(Protocol):
    """Structural type for the one method this module needs from an LLM client."""

    async def extract_claims_batch(self, tagged_paragraph: str) -> ParagraphClaimCollection: ...


async def extract_claims_for_document(
    llm_client: ClaimLLMClient,
    citations: List[ExtractedCitation],
    body_text: str,
    max_concurrency: int = 5,
) -> List[Claim]:
    """End-to-end: body markdown -> List[Claim], linked to `citations` by ref_id."""
    if not body_text or not body_text.strip() or not citations:
        return []

    style = detect_citation_style(body_text)
    if style == "unknown":
        return []

    tagged_paragraphs = _prepare_tagged_paragraphs(citations, body_text, style)
    if not tagged_paragraphs:
        return []

    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _run(tagged_text: str):
        async with sem:
            try:
                return await llm_client.extract_claims_batch(tagged_text)
            except Exception as exc:  # noqa: BLE001 - log and continue with other paragraphs
                print(f"[ClaimExtraction] Paragraph failed: {exc}")
                return None

    results = await asyncio.gather(*[_run(tagged_text) for tagged_text, _ in tagged_paragraphs])

    known_ref_ids = {c.ref_id for c in citations}
    return _assemble_claims(tagged_paragraphs, results, known_ref_ids)


def _prepare_tagged_paragraphs(
    citations: List[ExtractedCitation],
    body_text: str,
    style: str,
) -> List[Tuple[str, set]]:
    """Split body into paragraphs, canonicalize citations, tag sentences.

    Returns a list of (tagged_paragraph_text, valid_sentence_ids) — one
    entry per paragraph worth sending to the LLM. Paragraphs with fewer
    than 2 sentences AND no resolved citation are skipped (cheap boilerplate
    unlikely to contain a citable claim).
    """
    paragraphs = split_paragraphs(body_text)
    prepared: List[Tuple[str, set]] = []
    sid_counter = 1
    for para in paragraphs:
        cited_text, _resolved = tag_citations(para, citations, style=style)
        tagged_text, id_map, sid_counter = tag_sentences(cited_text, sid_counter)
        if not id_map:
            continue
        has_citation = "<CIT:" in tagged_text
        if not has_citation and len(id_map) < 2:
            continue
        valid_sids = {sid for sid, _ in id_map}
        prepared.append((tagged_text, valid_sids))
    return prepared


def _assemble_claims(
    tagged_paragraphs: List[Tuple[str, set]],
    results: List,
    known_ref_ids: set,
) -> List[Claim]:
    claims: List[Claim] = []
    claim_counter = 1
    for (_, valid_sids), result in zip(tagged_paragraphs, results):
        if result is None:
            continue
        for pc in result.claims:
            claim_text = (pc.claim or "").strip()
            if not claim_text:
                continue

            explicit = [r for r in dict.fromkeys(pc.explicit_citations) if r in known_ref_ids]
            inherited = [
                r
                for r in dict.fromkeys(pc.inherited_citations)
                if r in known_ref_ids and r not in explicit
            ]
            references = list(dict.fromkeys(explicit + inherited))
            source_ids = [s for s in dict.fromkeys(pc.source_sentence_ids) if s in valid_sids]

            claims.append(
                Claim(
                    claim_id=f"claim_{claim_counter:03d}",
                    claim=claim_text,
                    source_sentence_ids=source_ids,
                    explicit_citations=explicit,
                    inherited_citations=inherited,
                    references=references,
                )
            )
            claim_counter += 1
    return claims


def attach_claim_ids(citations: List[ExtractedCitation], claims: List[Claim]) -> None:
    """Populate `ExtractedCitation.claim_ids` in place from `claims`."""
    by_ref = {c.ref_id: c for c in citations}
    for claim in claims:
        for ref_id in claim.references:
            citation = by_ref.get(ref_id)
            if citation is not None and claim.claim_id not in citation.claim_ids:
                citation.claim_ids.append(claim.claim_id)
