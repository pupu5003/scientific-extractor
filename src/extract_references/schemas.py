"""
schemas.py
Defines the strict data contracts for the extraction pipeline.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re


class ExtractedIdentifiers(BaseModel):
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID (e.g., 2402.12345)")
    url: Optional[str] = Field(default=None, description="Fallback URL")


class ExtractedCitation(BaseModel):
    ref_id: str = Field(..., description="Local, canonical reference ID (e.g., R1). This is the "
                                          "system's own internal ID — it is NOT guaranteed to match "
                                          "any bracket number printed in the source PDF.")
    raw_text: Optional[str] = Field(default=None, description="The original raw text from the PDF")

    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list, description="List of 'Given Family' strings")
    venue: Optional[str] = None
    year: Optional[int] = None
    identifiers: ExtractedIdentifiers = Field(default_factory=ExtractedIdentifiers)
    claim_ids: List[str] = Field(
        default_factory=list,
        description="IDs of entries in the top-level `claims` list that cite this reference "
                    "(either explicitly or via inherited/discourse citation).",
    )

    @field_validator("year", mode="before")
    @classmethod
    def parse_year(cls, v):
        if not v:
            return None
        m = re.search(r"(1[89]\d{2}|20\d{2})", str(v))
        return int(m.group(1)) if m else None


class ParsedCitationEntry(BaseModel):
    """A single citation parsed by the LLM in a single-pass batch extraction."""
    raw_text: Optional[str] = Field(
        default=None,
        description="Verbatim text of this citation as it appeared in the source document",
    )
    title: str = Field(..., description="Full title of the paper")
    authors: List[str] = Field(
        default_factory=list,
        description="List of author names in 'FirstName LastName' format",
    )
    venue: Optional[str] = Field(None, description="Journal name, conference, or repository")
    year: Optional[int] = Field(None, description="4-digit publication year")
    doi: Optional[str] = Field(None, description="DOI if present in the source text")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID if present (e.g. 2402.12345)")
    url: Optional[str] = Field(None, description="URL if present and no DOI/arXiv ID")


class CitationCollection(BaseModel):
    """Validated array of citations produced from a single-pass LLM call."""
    citations: List[ParsedCitationEntry] = Field(
        ...,
        description="One entry per unique paper found in the references block",
    )


# ---------------------------------------------------------------------------
# Claim extraction (paragraph-level, citation-aligned)
# ---------------------------------------------------------------------------

class ParagraphClaim(BaseModel):
    """One claim as returned by the LLM for a single paragraph.

    `explicit_citations` / `inherited_citations` must only ever contain ref
    ids that were literally tagged as '<CIT:ref_id>' somewhere in that
    paragraph — this is enforced again downstream (anti-hallucination) in
    `claim_extraction.py`, but the model is instructed accordingly too.
    """
    claim: str = Field(..., description="Self-contained statement of the claim")
    source_sentence_ids: List[str] = Field(
        default_factory=list,
        description="The [S<n>] sentence id(s) this claim is drawn from",
    )
    explicit_citations: List[str] = Field(
        default_factory=list,
        description="Ref ids whose <CIT:...> tag appears directly in the claim's own sentence(s)",
    )
    inherited_citations: List[str] = Field(
        default_factory=list,
        description="Ref ids understood only via discourse reference to an earlier "
                    "sentence in the same paragraph, not tagged in the claim's own sentence(s)",
    )


class ParagraphClaimCollection(BaseModel):
    """Validated array of claims produced from a single-paragraph LLM call."""
    claims: List[ParagraphClaim] = Field(default_factory=list)


class Claim(BaseModel):
    """A final, document-level claim with a stable id and resolved reference ids."""
    claim_id: str = Field(..., description="Stable id for this claim, e.g. 'claim_001'")
    claim: str
    source_sentence_ids: List[str] = Field(default_factory=list)
    explicit_citations: List[str] = Field(default_factory=list)
    inherited_citations: List[str] = Field(default_factory=list)
    references: List[str] = Field(
        default_factory=list,
        description="Deduplicated union of explicit_citations + inherited_citations",
    )


class ExtractionResult(BaseModel):
    """Top-level pipeline output: the reference list plus the claims that cite them."""
    references: List[ExtractedCitation] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
