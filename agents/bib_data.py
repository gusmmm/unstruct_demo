
"""
Google GenAI agent to extract bibliographic metadata from Unstructured JSON elements.

Reads an elements JSON (e.g., files/hydrocortisone/hydrocortisone-output.json),
builds a concise first-page context, asks Gemini to return structured
ArticleMetadata JSON, validates it, and writes the result alongside the input file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bib_data")


class ArticleMetadata(BaseModel):
    """Pydantic model for journal article metadata."""

    title: Optional[str] = Field(None, description="Article title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    journal: Optional[str] = Field(None, description="Journal name")
    volume: Optional[str] = Field(None, description="Volume number")
    issue: Optional[str] = Field(None, description="Issue number")
    pages: Optional[str] = Field(None, description="Page range")
    year: Optional[int] = Field(None, description="Publication year")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    issn: Optional[str] = Field(None, description="ISSN")
    abstract: Optional[str] = Field(None, description="Article abstract")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    article_type: Optional[str] = Field("research_article", description="Type of article")
    publisher: Optional[str] = Field(None, description="Publisher name")
    url: Optional[str] = Field(None, description="Article URL")
    citation_info: Optional[str] = Field(None, description="Full citation information")
    extracted_date: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat(), description="Extraction timestamp")

    # Note: Avoid Pydantic validators; Gemini structured output currently ignores validators
    # and .parsed may be empty if ValidationError occurs (per docs).


def ensure_env_var(key: str) -> None:
    """Ensure an env var exists, create placeholder in .env if missing."""
    load_dotenv(override=False)
    if os.getenv(key):
        return
    env_path = Path(".env")
    placeholder = f"{key}=YOUR_API_KEY_HERE\n"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        if key not in content:
            with env_path.open("a", encoding="utf-8") as f:
                f.write(placeholder)
            logger.warning("%s not set. Added placeholder to .env", key)
    else:
        env_path.write_text(placeholder, encoding="utf-8")
        logger.warning("Created .env with placeholder for %s", key)


def load_elements(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Elements JSON must be a list")
    return data


def get_first_page_number(elements: List[Dict[str, Any]]) -> Optional[int]:
    """Try to find an explicit page number for the first element.

    Not all exports carry page_number. We'll also support a PageBreak-based fallback.
    """
    for el in elements:
        md = el.get("metadata") or {}
        pn = md.get("page_number") or md.get("page_number_0")
        if isinstance(pn, int):
            return pn
    return None


def collect_first_page_elements(elements: List[Dict[str, Any]], max_total_chars: int = 8000) -> List[Dict[str, str]]:
    """Return ordered list of {type, text} for page 1.

    Strategy:
    - Prefer elements where metadata.page_number == first_page.
    - If page numbers are absent, take elements from the start up to the first PageBreak.
    - Include only entries with non-empty text; keep type label for better model grounding.
    - Cap total accumulated characters to avoid overly long prompts.
    """
    page1: List[Dict[str, str]] = []
    total = 0
    first_page = get_first_page_number(elements)
    if first_page is not None:
        for el in elements:
            md = el.get("metadata") or {}
            if md.get("page_number") != first_page:
                continue
            t = el.get("text") or ""
            typ = el.get("type") or ""
            if isinstance(t, str) and t.strip():
                entry = {"type": str(typ), "text": t.strip()}
                total += len(entry["text"]) + 16
                page1.append(entry)
            if total >= max_total_chars:
                break
    else:
        for el in elements:
            typ = el.get("type") or ""
            if typ == "PageBreak":
                break
            t = el.get("text") or ""
            if isinstance(t, str) and t.strip():
                entry = {"type": str(typ), "text": t.strip()}
                total += len(entry["text"]) + 16
                page1.append(entry)
            if total >= max_total_chars:
                break
    return page1


def build_prompt(page1_items: List[Dict[str, str]], extra_hints: Optional[str] = None) -> List[Dict[str, Any]]:
    """Create parts for the GenAI prompt using ordered {type,text} entries from page 1."""
    instructions = (
        "You are an expert academic librarian and metadata extraction specialist.\n"
        "Use the ordered first-page elements (type + text) to extract the article's bibliographic metadata.\n"
        "Fill unknown fields with null or []. Respond only with the structured data per the configured schema."
    )
    # Serialize first-page items compactly
    items_text = json.dumps(page1_items, ensure_ascii=False, indent=2)
    parts: List[Dict[str, Any]] = [
        {"text": instructions},
        {"text": "First page elements (ordered):"},
        {"text": items_text},
    ]
    if extra_hints:
        parts.append({"text": extra_hints})
    return parts

def build_schema_dict() -> Dict[str, Any]:
    """Return a permissive dict-based schema for ArticleMetadata for fallback."""
    return {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING", "nullable": True},
            "authors": {"type": "ARRAY", "items": {"type": "STRING"}},
            "journal": {"type": "STRING", "nullable": True},
            "volume": {"type": "STRING", "nullable": True},
            "issue": {"type": "STRING", "nullable": True},
            "pages": {"type": "STRING", "nullable": True},
            "year": {"type": "INTEGER", "nullable": True},
            "doi": {"type": "STRING", "nullable": True},
            "pmid": {"type": "STRING", "nullable": True},
            "issn": {"type": "STRING", "nullable": True},
            "abstract": {"type": "STRING", "nullable": True},
            "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
            "article_type": {"type": "STRING", "nullable": True},
            "publisher": {"type": "STRING", "nullable": True},
            "url": {"type": "STRING", "nullable": True},
            "citation_info": {"type": "STRING", "nullable": True},
            "extracted_date": {"type": "STRING", "nullable": True},
        },
    }


def _sanitize_article_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and coerce model output into ArticleMetadata-compatible dict.

    - Only keep known fields.
    - Coerce authors/keywords to list[str].
    - Coerce year to int when possible.
    - Normalize DOI text.
    - Ensure extracted_date is a string timestamp.
    """
    allowed = set(ArticleMetadata.model_fields.keys())
    cleaned: Dict[str, Any] = {k: v for k, v in (data or {}).items() if k in allowed}

    # Authors
    authors = cleaned.get("authors")
    if isinstance(authors, str):
        parts = [a.strip() for a in authors.replace(";", ",").split(",")]
        cleaned["authors"] = [p for p in parts if p]
    elif isinstance(authors, list):
        cleaned["authors"] = [str(a).strip() for a in authors if str(a).strip()]

    # Keywords
    keywords = cleaned.get("keywords")
    if isinstance(keywords, str):
        parts = [k.strip() for k in keywords.replace(";", ",").split(",")]
        cleaned["keywords"] = [p for p in parts if p]
    elif isinstance(keywords, list):
        cleaned["keywords"] = [str(k).strip() for k in keywords if str(k).strip()]

    # Year
    year = cleaned.get("year")
    if isinstance(year, str):
        try:
            cleaned["year"] = int("".join(ch for ch in year if ch.isdigit())[:4])
        except Exception:
            cleaned["year"] = None

    # DOI
    doi = cleaned.get("doi")
    if isinstance(doi, str):
        s = doi.strip()
        # If a full URL was provided, try to extract the DOI path after doi.org/
        m = re.search(r"doi\.org/([^\s]+)", s, flags=re.IGNORECASE)
        if m:
            s = m.group(1)
        # Strip common prefixes
        for prefix in ("doi:", "DOI:", "DOI", "doi"):
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):].strip(" :/")
        # Normalize unicode dashes to ascii hyphen
        s = s.replace("—", "-").replace("–", "-")
        # Replace internal whitespace with hyphens (observed in PDFs splitting DOI chunks)
        s = re.sub(r"\s+", "-", s)
        # Trim trailing punctuation that might get captured
        s = s.strip(" .,),;]")
        cleaned["doi"] = s or None

    # Canonical URL from DOI (override any malformed URL)
    if cleaned.get("doi"):
        cleaned["url"] = f"https://doi.org/{cleaned['doi']}"

    # Pages: join numeric range with hyphen
    pages = cleaned.get("pages")
    if isinstance(pages, str):
        nums = re.findall(r"\d+", pages)
        if len(nums) == 2:
            cleaned["pages"] = f"{nums[0]}-{nums[1]}"

    # extracted_date
    ed = cleaned.get("extracted_date")
    if not isinstance(ed, str) or not ed.strip():
        cleaned["extracted_date"] = datetime.now().isoformat()

    return cleaned

def call_genai_extract(
    client: genai.Client,
    model: str,
    parts: List[Dict[str, Any]],
    max_tokens: int = 2000,
) -> ArticleMetadata:
    # First attempt: Pydantic response schema
    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": parts}],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ArticleMetadata,
            max_output_tokens=max_tokens,
            temperature=0.1,
        ),
    )
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        logger.warning("Structured output parse (Pydantic) returned None; retrying with dict schema.")
        # Second attempt: dict-based schema
        resp2 = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": parts}],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=build_schema_dict(),
                max_output_tokens=max_tokens,
                temperature=0.1,
            ),
        )
        parsed2 = getattr(resp2, "parsed", None)
        if isinstance(parsed2, dict):
            return ArticleMetadata(**_sanitize_article_dict(parsed2))
        if isinstance(parsed2, list) and parsed2 and isinstance(parsed2[0], dict):
            return ArticleMetadata(**_sanitize_article_dict(parsed2[0]))
        # Fall back to response.text JSON parsing
        text = getattr(resp2, "text", None) or getattr(resp2, "output_text", None)
        if isinstance(text, str) and text.strip():
            try:
                data = json.loads(text)
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    return ArticleMetadata(**_sanitize_article_dict(data))
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to load JSON from fallback text: %s", e)
        raise RuntimeError("Model returned no parsed data after fallback attempts.")
    # If the SDK returns a dict instead of a Pydantic model, coerce it
    if isinstance(parsed, ArticleMetadata):
        return parsed
    if isinstance(parsed, dict):
        return ArticleMetadata(**_sanitize_article_dict(parsed))
    # Some SDK versions may return a list with a single item
    if isinstance(parsed, list) and parsed:
        item = parsed[0]
        if isinstance(item, ArticleMetadata):
            return item
        if isinstance(item, dict):
            return ArticleMetadata(**_sanitize_article_dict(item))
    raise RuntimeError(f"Unexpected parsed type: {type(parsed)!r}")


def write_output(meta: ArticleMetadata, input_path: Path, out_path: Optional[Path]) -> Path:
    if out_path is None:
        # files/hydrocortisone/hydrocortisone-output.json -> files/hydrocortisone/hydrocortisone-bibliography.json
        stem = input_path.stem.replace("-output", "")
        name = f"{stem}-bibliography.json"
        out_path = input_path.parent / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Final sanitize before writing to guarantee consistent formatting
    payload = _sanitize_article_dict(meta.model_dump())
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract bibliographic metadata via Google GenAI")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("./files/hydrocortisone/hydrocortisone-output.json"),
        help="Path to Unstructured elements JSON",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    p.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model name")
    p.add_argument("--max-tokens", type=int, default=600, help="Max output tokens")
    p.add_argument("--dry-run", action="store_true", help="Skip GenAI call and only build context")
    return p.parse_args()


def main() -> None:
    # Env & client
    ensure_env_var("GEMINI_API_KEY")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")

    args = parse_args()
    elements = load_elements(args.input)

    logger.info("Loaded %d elements from %s", len(elements), args.input)

    page1_items = collect_first_page_elements(elements)
    if not page1_items:
        raise RuntimeError("No first-page elements found to build prompt")

    if args.dry_run:
        preview_chars = sum(len(it.get("text", "")) for it in page1_items)
        logger.info("Dry run: built %d first-page items (%d chars). Skipping GenAI call.", len(page1_items), preview_chars)
        # Write a minimal skeleton so downstream steps can proceed
        meta = ArticleMetadata()
        out = write_output(meta, args.input, args.out)
        logger.info("Wrote skeleton metadata to %s", out)
        return

    client = genai.Client(api_key=api_key)
    parts = build_prompt(page1_items)
    logger.info("Calling GenAI model (structured output): %s", args.model)
    meta = call_genai_extract(client, args.model, parts, max_tokens=args.max_tokens)

    out_path = write_output(meta, args.input, args.out)
    logger.info("Wrote metadata JSON to %s", out_path)


if __name__ == "__main__":
    main()

    
    