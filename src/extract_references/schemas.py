"""
schemas.py
Defines the strict data contracts for the extraction pipeline.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import re

class ExtractedIdentifiers(BaseModel):
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID (e.g., 2402.12345)")
    url: Optional[str] = Field(default=None, description="Fallback URL")

class ExtractedCitation(BaseModel):
    ref_id: str = Field(..., description="Local reference ID (e.g., R1)")
    raw_text: str = Field(..., description="The original raw text from the PDF")
    
    # Critical Metadata
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list, description="List of 'Given Family' strings")
    venue: Optional[str] = None
    year: Optional[int] = None
    identifiers: ExtractedIdentifiers = Field(default_factory=ExtractedIdentifiers)
    
    # Provenance tracking
    confidence_score: float = Field(default=1.0, description="1.0 if deterministic, lower if LLM patched")
    llm_reviewed: bool = Field(default=False)

    @field_validator('year', mode='before')
    def parse_year(cls, v):
        if not v:
            return None
        m = re.search(r"(1[89]\d{2}|20\d{2})", str(v))
        return int(m.group(1)) if m else None

class LLMPatchInstruction(BaseModel):
    """Structured output expected from the LLM"""
    fill: Dict[str, Any] = Field(default_factory=dict, description="Fields to add if currently null")
    corrections: Dict[str, Any] = Field(default_factory=dict, description="Fields to overwrite if incorrect")