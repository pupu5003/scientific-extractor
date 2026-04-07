"""
heuristics.py
Minimal heuristic filter to drop non-reference artifacts from extraction output.
"""
from typing import Dict, Any

class CitationParserEngine:

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