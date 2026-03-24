"""
heuristics.py
Deterministic parsing rules, anystyle JSON digestion, and anti-hallucination guards.
"""
import re
from typing import Dict, Any, List, Optional

class CitationParserEngine:

    @staticmethod
    def digest_anystyle_json(raw_text: str, anystyle_result: dict) -> Dict[str, Any]:
        """Maps anystyle CLI JSON output into a flat internal dict."""
        result: Dict[str, Any] = {"raw_text": raw_text}
        if not anystyle_result:
            return result

        # Title — anystyle returns list of strings
        titles = anystyle_result.get("title", [])
        if titles:
            result["title"] = titles[0] if isinstance(titles[0], str) else str(titles[0])

        # Authors — list of dicts with optional 'given' / 'family', or plain strings
        authors: List[str] = []
        for a in anystyle_result.get("author", []):
            if isinstance(a, dict):
                given = (a.get("given") or "").strip()
                family = (a.get("family") or "").strip()
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            elif isinstance(a, str) and a.strip():
                authors.append(a.strip())
        if authors:
            result["authors"] = authors

        # Year — anystyle 'date' is a list of strings like ["2015"]
        dates = anystyle_result.get("date", [])
        if dates:
            result["year"] = str(dates[0])

        # Venue — prefer journal, then booktitle/container-title/publisher
        for venue_key in ("journal", "booktitle", "container-title", "publisher", "series"):
            values = anystyle_result.get(venue_key, [])
            if values:
                result["venue"] = values[0] if isinstance(values[0], str) else str(values[0])
                break

        # Identifiers
        dois = anystyle_result.get("doi", [])
        if dois:
            result["doi"] = dois[0] if isinstance(dois[0], str) else str(dois[0])

        urls = anystyle_result.get("url", [])
        if urls:
            result["url"] = urls[0] if isinstance(urls[0], str) else str(urls[0])

        return result

    @staticmethod
    def apply_regex_fallbacks(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Applies high-confidence regex to fill gaps left by GROBID."""
        raw = parsed.get("raw_text", "")
        
        if not parsed.get("doi"):
            m = re.search(r"(10\.\d{4,}/[^\s,;]+)", raw)
            if m: parsed["doi"] = m.group(1).rstrip(".")
                
        if not parsed.get("arxiv_id"):
            m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", raw)
            if m: parsed["arxiv_id"] = m.group(1)
            
        if not parsed.get("url"):
            m = re.search(r"https?://[^\s]+(?:\s+[^\s,\)\.]+)*", raw, re.IGNORECASE)
            if m: parsed["url"] = re.sub(r"\s+", "", m.group(0)).rstrip(".,)")
            
        if not parsed.get("year"):
            m = re.search(r"\b(1[89]\d{2}|20\d{2})(?:[a-z])?\b", raw, re.IGNORECASE)
            if m: parsed["year"] = m.group(1)

        regex_authors = CitationParserEngine._extract_authors_from_raw(raw)
        if regex_authors:
            parsed["authors_regex"] = regex_authors

        parsed["authors"] = CitationParserEngine._merge_authors(
            parsed.get("authors"),
            parsed.get("authors_regex"),
            raw,
        )

        return parsed

    @staticmethod
    def _extract_authors_from_raw(raw: str) -> List[str]:
        """Heuristic regex-only to extract authors from the head of a citation."""
        head = raw

        year_block = re.search(r"\.\s*(1[89]\d{2}|20\d{2})[a-z]?\.\s+", raw, re.IGNORECASE)
        if year_block:
            head = raw[:year_block.start() + 1]
        else:
            end_m = re.search(r"(?<=[a-zà-ỹ])\.\s+[A-Z]", raw)
            if end_m:
                head = raw[: end_m.start() + 1]
            else:
                year_m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", raw)
                if year_m:
                    head = raw[:year_m.start()]

        head = re.sub(r"\bet\s+al\.?", "", head, flags=re.IGNORECASE)
        head = re.sub(r"\band\b", ",", head, flags=re.IGNORECASE)
        head = re.sub(r"\s+", " ", head).strip(" .,")

        parts = [p.strip(" .,") for p in head.split(",") if p.strip(" .,")]
        authors: List[str] = []
        for p in parts:
            if len(p) < 2:
                continue
            if '"' in p or "'" in p and len(p.split()) > 4:
                continue
            if re.search(r"\b(arxiv|proceedings|press|journal|conference|vol\.|pp\.|edition)\b", p, re.IGNORECASE):
                continue
            if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|june)\b", p, re.IGNORECASE):
                continue
            if re.search(r"\d", p):
                continue
            if len(p.split()) < 2:
                continue
            authors.append(p)

        return authors

    @staticmethod
    def _merge_authors(
        xml_authors: Optional[List[str]],
        regex_authors: Optional[List[str]],
        raw_text: str,
    ) -> List[str]:
        """Combine XML and regex author lists, preferring the more plausible option."""
        xml_authors = xml_authors or []
        regex_authors = regex_authors or []

        def score(authors: List[str]) -> int:
            if not authors:
                return 0
            single_tokens = sum(1 for a in authors if len(a.split()) < 2)
            hyphen_fragments = sum(1 for a in authors if a.startswith("-"))
            return len(authors) * 3 - single_tokens * 4 - hyphen_fragments * 2

        xml_score = score(xml_authors)
        regex_score = score(regex_authors)

        if regex_score > xml_score:
            return regex_authors

        if xml_authors:
            return xml_authors

        return regex_authors

    @staticmethod
    def is_plausible_reference(raw_text: str, parsed: Dict[str, Any]) -> bool:
        """Heuristic filter to drop non-reference artifacts from GROBID output."""
        if not raw_text or len(raw_text.strip()) < 10:
            return False

        fields_present = 0
        if parsed.get("authors"):
            fields_present += 1
        if parsed.get("title"):
            fields_present += 1
        if parsed.get("venue"):
            fields_present += 1
        if parsed.get("year"):
            fields_present += 1
        if parsed.get("doi") or parsed.get("arxiv_id") or parsed.get("url"):
            fields_present += 1

        return fields_present >= 2

    @staticmethod
    def guard_hallucinations(raw_text: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Strips exact-match fields (URLs, DOIs) from LLM output if they don't exist in raw_text."""
        safe_patch = {}
        exact_fields = {"doi", "arxiv_id", "url"}
        for k, v in patch.items():
            if k in exact_fields and isinstance(v, str):
                # Clean strings for loose matching
                if v.strip().lower() not in raw_text.lower().replace(" ", ""):
                    continue # Hallucination detected
            safe_patch[k] = v
        return safe_patch