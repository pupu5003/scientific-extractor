"""
clients.py
Async HTTP clients with fault tolerance for GROBID and LLM providers.
"""
import aiohttp
import asyncio
import json
import re
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from .schemas import LLMPatchInstruction

class AsyncGrobidClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def extract_raw_references(self, pdf_path: str) -> List[str]:
        """Uploads PDF and extracts raw citation strings."""
        url = f"{self.base_url}/api/processReferences"
        data = aiohttp.FormData()
        data.add_field('input', open(pdf_path, 'rb'), filename='paper.pdf', content_type='application/pdf')
        data.add_field('consolidateCitations', '0')
        data.add_field('includeRawCitations', '1')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, timeout=120) as response:
                response.raise_for_status()
                xml_content = await response.text()
                return self._parse_tei_xml_for_raw(xml_content)

    def _parse_tei_xml_for_raw(self, xml_content: str) -> List[str]:
        soup = BeautifulSoup(xml_content, "xml")
        references = []
        for bibl in soup.find_all("biblStruct"):
            note = bibl.find("note", type="raw_reference")
            if note and note.get_text(strip=True):
                # Clean up hyphenation from PDF line breaks
                clean_text = note.get_text(strip=True).replace("- ", "")
                # Strip leading citation labels like "ACG + 16]"
                clean_text = re.sub(r"^[A-Z]{2,}\s*\+\s*\d{2}\]\s*", "", clean_text)
                references.append(clean_text)
        return references



class AnystyleClient:
    """Calls the local anystyle CLI to parse a single citation string into a structured dict."""

    async def parse(self, raw_text: str) -> Dict[str, Any]:
        """Write raw_text to a temp file, call anystyle CLI, return first parsed record."""
        import tempfile, os
        tmp = None
        try:
            # anystyle CLI on macOS does not support stdin ('-'), so use a temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                             delete=False, encoding="utf-8") as f:
                f.write(raw_text)
                tmp = f.name

            proc = await asyncio.create_subprocess_exec(
                "anystyle", "-f", "json", "parse", tmp,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=15,
            )
            if proc.returncode != 0:
                return {}
            records = json.loads(stdout.decode("utf-8"))
            if isinstance(records, list) and records:
                return records[0]
            return {}
        except (asyncio.TimeoutError, FileNotFoundError, json.JSONDecodeError, Exception):
            return {}
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)


class AsyncLLMClient:
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def review_citation(self, raw_text: str, current_parsed: dict) -> LLMPatchInstruction:
        """Asks the LLM to patch missing or incorrect fields based strictly on raw_text."""
        system_prompt = (
            "You are a strict Data Extraction Agent. Source of truth is ONLY the `raw_text`.\n"
            "Return JSON exactly matching the requested schema: {\"fill\": {}, \"corrections\": {}}.\n"
            "Do not invent DOIs, URLs, or Authors."
        )
        user_prompt = f"raw_text: {raw_text}\ncurrently_parsed: {json.dumps(current_parsed)}"

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        
        content = response.choices[0].message.content
        return LLMPatchInstruction.model_validate_json(content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def split_citations(self, raw_text: str) -> List[str]:
        """Asks the LLM to split a merged citation string into individual citations."""
        system_prompt = (
            "You are a citation segmentation expert. Some bibliographic references are merged into a single string.\n"
            "Split them into individual, complete citations. A new citation usually starts with a list of authors.\n"
            "STRICT RULES:\n"
            "1. NO HALLUCINATION: Do not add any information (years, DOIs, authors) that is not in the input.\n"
            "2. EXACT TEXT: Each output string must be a direct subset of the input text.\n"
            "3. Return a JSON list of strings."
            "\nReturn JSON format: {\"citations\": [\"citation 1\", \"citation 2\", ...]}"
        )
        user_prompt = f"raw_text to split: {raw_text}"

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        
        try:
            content = json.loads(response.choices[0].message.content)
            return content.get("citations", [raw_text])
        except Exception:
            return [raw_text]