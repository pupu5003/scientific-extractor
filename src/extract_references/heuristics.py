"""
heuristics.py
Deterministic parsing rules and anti-hallucination guards.
"""
import re
from typing import Dict, Any, List, Optional

class CitationParserEngine:

    @staticmethod
    def heal_broken_urls(text: str) -> str:
        """Fixes common URL/DOI fragmentation caused by PDF line breaks (e.g. 'https : //')."""
        if not text:
            return text
        
        # 1. First, fix the core scheme and separators (e.g. https : / / -> https://)
        text = re.sub(r'(https?)\s*[:\s/]+', r'\1://', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDOI\s*[:\s]+10\s*\.\s*', r'DOI:10.', text, flags=re.IGNORECASE)

        # 2. Aggressive join for suspected URL blocks
        def fix_url_block(m):
            return re.sub(r'\s+', '', m.group(0))

        # Regex for likely URL block: http://... or 10.xxxx/...
        # Matches alphanumeric and common URL punctuation, including spaces between them
        # Stopper: Stop if we find a space followed by a common non-URL word or start of title
        url_pattern = r'https?://[a-zA-Z0-9\.\/_:#%?=+-]+(?:\s+[a-zA-Z0-9\.\/_:#%?=+-]+)*'
        doi_pattern = r'10\.\d{4,}/[a-zA-Z0-9\.\/_:#%?=+-]+(?:\s+[a-zA-Z0-9\.\/_:#%?=+-]+)*'
        
        text = re.sub(url_pattern, fix_url_block, text, flags=re.IGNORECASE)
        text = re.sub(doi_pattern, fix_url_block, text)
        
        return text.strip()

    @staticmethod
    def is_plausible_reference(raw_text: str, parsed: Dict[str, Any]) -> bool:
        """Heuristic filter to drop non-reference artifacts from extraction output."""
        if not raw_text or len(raw_text.strip()) < 10:
            return False

        # Strong signal: If we have an identifier AND authors, it's a reference
        # even if extraction failed to correctly identify a 'title' (common in long author lists).
        has_strong_id = bool(parsed.get("doi") or parsed.get("arxiv_id") or parsed.get("url"))
        has_authors = bool(parsed.get("authors"))
        has_year = bool(parsed.get("year"))
        has_title = bool(parsed.get("title"))

        # Case 1: Standard reference with a title
        if has_title:
            # We still want at least one other field to be sure it's not a random sentence
            fields_present = 0
            if has_authors: fields_present += 1
            if has_title:   fields_present += 1
            if parsed.get("venue"): fields_present += 1
            if has_year:    fields_present += 1
            if has_strong_id: fields_present += 1
            return fields_present >= 2

        # Case 2: No title found, but strong ID + authors/year exists
        if has_strong_id and (has_authors or has_year):
            return True

        return False
