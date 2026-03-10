"""
heuristics.py
Deterministic parsing rules, XML digestion, and anti-hallucination guards.
"""
import re
from bs4 import BeautifulSoup
from typing import Dict, Any, List

class CitationParserEngine:
    
    @staticmethod
    def digest_grobid_xml(raw_text: str, xml_content: str) -> Dict[str, Any]:
        """Converts GROBID TEI XML into a flat dictionary."""
        result = {"raw_text": raw_text}
        if not xml_content:
            return result
        
        soup = BeautifulSoup(xml_content, "xml")
        
        # Title
        title_node = soup.find("title", level="a") or soup.find("title", level="m")
        if title_node:
            result["title"] = title_node.get_text(strip=True)
            
        # Authors
        authors = []
        for author in soup.find_all("author"):
            pers = author.find("persName")
            if pers:
                given = " ".join([f.get_text(strip=True) for f in pers.find_all("forename")])
                surname = pers.find("surname")
                surname = surname.get_text(strip=True) if surname else ""
                if given or surname:
                    authors.append(f"{given} {surname}".strip())
        if authors:
            result["authors"] = authors

        # Year
        date_node = soup.find("date")
        if date_node and date_node.get("when"):
            result["year"] = date_node.get("when")
            
        # Venue (Journal/Conf/Proceedings)
        venue_node = soup.find("title", level="j") or soup.find("title", level="m")
        if venue_node and venue_node != title_node:
            result["venue"] = venue_node.get_text(strip=True)
            
        # Identifiers
        for idno in soup.find_all("idno"):
            t = str(idno.get("type") or "").strip().lower()
            val = idno.get_text(strip=True)
            if t == "doi": result["doi"] = val
            elif "arxiv" in t: result["arxiv_id"] = val
            
        # URLs from pointers
        ptr = soup.find("ptr", target=True)
        if ptr and not result.get("url"):
            result["url"] = ptr["target"]
            
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

        return parsed

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