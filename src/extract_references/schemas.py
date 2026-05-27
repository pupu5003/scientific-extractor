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
    ref_id: str = Field(..., description="Local reference ID (e.g., R1)")
    raw_text: Optional[str] = Field(default=None, description="The original raw text from the PDF")

    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list, description="List of 'Given Family' strings")
    venue: Optional[str] = None
    year: Optional[int] = None
    identifiers: ExtractedIdentifiers = Field(default_factory=ExtractedIdentifiers)

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