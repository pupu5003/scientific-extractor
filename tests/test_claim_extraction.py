"""
Unit tests for src/extract_references/claim_extraction.py.

No real LLM call is made — a FakeLLMClient returns scripted
ParagraphClaimCollection objects so the orchestration/assembly logic
(anti-hallucination filtering, claim id assignment, attach_claim_ids) can be
verified deterministically.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract_references.schemas import (
    ExtractedCitation,
    ParagraphClaim,
    ParagraphClaimCollection,
)
from src.extract_references.claim_extraction import (
    extract_claims_for_document,
    attach_claim_ids,
    _prepare_tagged_paragraphs,
)


def make_citation(ref_id, surname, year, first_given="Jane"):
    return ExtractedCitation(
        ref_id=ref_id,
        raw_text=f"{first_given} {surname}. Title. Venue, {year}.",
        authors=[f"{first_given} {surname}"],
        year=year,
    )


BODY = (
    "To improve their reliability, LLMs rely on Chain-of-thoughts (CoT) "
    "(Wei et al., 2022) and Test-Time Compute (TTC) (Snell et al., 2024). "
    "CoTs seek to emulate human reasoning by having the model generate "
    "step-by-step reasoning traces before producing an answer."
)


class FakeLLMClient:
    """Returns one scripted ParagraphClaimCollection per call, in order."""

    def __init__(self, scripted_results):
        self._results = list(scripted_results)
        self.calls = []

    async def extract_claims_batch(self, tagged_paragraph: str) -> ParagraphClaimCollection:
        self.calls.append(tagged_paragraph)
        return self._results.pop(0)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# _prepare_tagged_paragraphs
# ---------------------------------------------------------------------------

def test_prepare_tagged_paragraphs_tags_citations_and_sentences():
    citations = [make_citation("R17", "Wei", 2022), make_citation("R42", "Snell", 2024)]
    prepared = _prepare_tagged_paragraphs(citations, BODY, style="author_year")
    assert len(prepared) == 1
    tagged_text, valid_sids = prepared[0]
    assert "<CIT:R17>" in tagged_text
    assert "<CIT:R42>" in tagged_text
    assert "[S1]" in tagged_text and "[S2]" in tagged_text
    assert valid_sids == {"S1", "S2"}


# ---------------------------------------------------------------------------
# extract_claims_for_document — the inherited-citation scenario from the spec
# ---------------------------------------------------------------------------

def test_explicit_and_inherited_citation_are_kept_separate():
    citations = [make_citation("R17", "Wei", 2022), make_citation("R42", "Snell", 2024)]

    scripted = ParagraphClaimCollection(
        claims=[
            ParagraphClaim(
                claim="Chain-of-thought reasoning can improve LLM reliability.",
                source_sentence_ids=["S1"],
                explicit_citations=["R17"],
                inherited_citations=[],
            ),
            ParagraphClaim(
                claim="Test-time compute can improve LLM reliability.",
                source_sentence_ids=["S1"],
                explicit_citations=["R42"],
                inherited_citations=[],
            ),
            ParagraphClaim(
                claim="CoT has the model generate step-by-step reasoning traces before answering.",
                source_sentence_ids=["S2"],
                explicit_citations=[],
                inherited_citations=["R17"],  # discourse link back to S1's <CIT:R17>
            ),
        ]
    )
    fake_client = FakeLLMClient([scripted])

    claims = run(extract_claims_for_document(fake_client, citations, BODY, max_concurrency=2))

    assert len(claims) == 3
    assert claims[0].claim_id == "claim_001"
    assert claims[1].claim_id == "claim_002"
    assert claims[2].claim_id == "claim_003"

    assert claims[2].explicit_citations == []
    assert claims[2].inherited_citations == ["R17"]
    assert claims[2].references == ["R17"]

    attach_claim_ids(citations, claims)
    r17 = next(c for c in citations if c.ref_id == "R17")
    r42 = next(c for c in citations if c.ref_id == "R42")
    assert r17.claim_ids == ["claim_001", "claim_003"]
    assert r42.claim_ids == ["claim_002"]


def test_hallucinated_ref_id_is_dropped():
    citations = [make_citation("R17", "Wei", 2022), make_citation("R42", "Snell", 2024)]
    scripted = ParagraphClaimCollection(
        claims=[
            ParagraphClaim(
                claim="Some claim.",
                source_sentence_ids=["S1"],
                explicit_citations=["R17", "R999"],  # R999 was never tagged in this paragraph
                inherited_citations=[],
            ),
        ]
    )
    fake_client = FakeLLMClient([scripted])
    claims = run(extract_claims_for_document(fake_client, citations, BODY, max_concurrency=2))
    assert claims[0].explicit_citations == ["R17"]
    assert claims[0].references == ["R17"]


def test_hallucinated_sentence_id_is_dropped():
    citations = [make_citation("R17", "Wei", 2022), make_citation("R42", "Snell", 2024)]
    scripted = ParagraphClaimCollection(
        claims=[
            ParagraphClaim(
                claim="Some claim.",
                source_sentence_ids=["S1", "S99"],  # S99 does not exist in this paragraph
                explicit_citations=["R17"],
                inherited_citations=[],
            ),
        ]
    )
    fake_client = FakeLLMClient([scripted])
    claims = run(extract_claims_for_document(fake_client, citations, BODY, max_concurrency=2))
    assert claims[0].source_sentence_ids == ["S1"]


def test_paragraph_failure_does_not_crash_whole_document():
    citations = [make_citation("R17", "Wei", 2022), make_citation("R42", "Snell", 2024)]

    class FlakyClient:
        async def extract_claims_batch(self, tagged_paragraph):
            raise RuntimeError("simulated API failure")

    claims = run(extract_claims_for_document(FlakyClient(), citations, BODY, max_concurrency=2))
    assert claims == []


def test_no_citations_returns_empty():
    claims = run(extract_claims_for_document(FakeLLMClient([]), [], BODY))
    assert claims == []


def test_unknown_style_skips_llm_call_entirely():
    citations = [make_citation("R17", "Wei", 2022)]
    fake_client = FakeLLMClient([])
    claims = run(
        extract_claims_for_document(fake_client, citations, "No citation markers in this text at all.")
    )
    assert claims == []
    assert fake_client.calls == []
