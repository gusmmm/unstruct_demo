"""
Summarize Table elements for RAG: read tables_llm_dump.json, send caption + table text
(+ optional truncated HTML and nearby associated text) to Google GenAI, and save a per-table
summary JSON.

Usage:
  uv run iterations/tables_summary.py \
	--input ./files/extracted/tables/tables_llm_dump.json \
	--out ./files/extracted/tables/tables_llm_summaries.json \
	--model gemini-2.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tables_summary")


def load_dump(path: Path) -> List[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("Input JSON must be a list of table entries")
	return data


def _truncate(text: Optional[str], limit: int) -> Optional[str]:
	if text is None:
		return None
	if limit <= 0:
		return None
	return text if len(text) <= limit else text[:limit]


def summarize_one(
	client: genai.Client,
	model: str,
	element_id: str,
	caption: Optional[str],
	table_text: Optional[str],
	table_html: Optional[str],
	associated_text: Optional[List[str]] = None,
	max_tokens: int = 250,
	html_max_chars: int = 0,
) -> str:
	"""Call Google GenAI (google-genai) to summarize one table with provided context.

	The prompt includes caption, textual table extraction, optional truncated HTML, and nearby
	associated text. Output is a concise summary limited to ~250 tokens by config.
	"""
	parts: List[Dict[str, Any]] = []

	instructions = (
		"You are assisting with a clinical research paper analysis. "
		"Summarize the key information from the table in 3-6 concise sentences. "
		"Capture the table's purpose, key variables, groups, and the most important values or trends. "
		"If the table reports effect sizes, odds ratios, confidence/credible intervals, or probabilities, include them. "
		"Avoid speculation. Use plain language and keep technical accuracy. Max 250 tokens."
	)
	parts.append({"text": instructions})

	ctx_lines: List[str] = []
	if caption:
		ctx_lines.append(f"Caption: {caption}")
	if table_text:
		ctx_lines.append("Table text (extracted):\n" + table_text)

	if associated_text:
		joined = "\n".join([s for s in associated_text if s])
		if joined.strip():
			ctx_lines.append("Nearby associated text:\n" + joined)

	if ctx_lines:
		parts.append({"text": "\n\n".join(ctx_lines)})

	html_snippet = _truncate(table_html, html_max_chars)
	if html_snippet:
		parts.append({"text": "Table HTML (truncated):\n" + html_snippet})

	# google-genai Client usage: models.generate_content with contents and config
	resp = client.models.generate_content(
		model=model,
		contents=[{"role": "user", "parts": parts}],
		config={
			"max_output_tokens": max_tokens,
			"temperature": 0.2,
		},
	)

	# Extract text from response
	text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
	if isinstance(text, str) and text.strip():
		return text.strip()
	cand = getattr(resp, "candidates", None)
	if cand:
		for c in cand:
			content = getattr(c, "content", None) or {}
			parts_out = getattr(content, "parts", None) or []
			for p in parts_out:
				if isinstance(p, dict):
					t = p.get("text")
					if t:
						return str(t).strip()
				else:
					t = getattr(p, "text", None)
					if t:
						return str(t).strip()
	return ""


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Summarize table elements via Google GenAI")
	p.add_argument("--input", type=Path, default=Path("./files/extracted/tables/tables_llm_dump.json"))
	p.add_argument(
		"--out",
		type=Path,
		default=Path("./files/extracted/tables/tables_llm_summaries.json"),
	)
	p.add_argument("--model", default="gemini-2.5-flash-lite")
	p.add_argument("--max-tokens", type=int, default=250)
	p.add_argument(
		"--html-max-chars",
		type=int,
		default=0,
		help="If > 0, include up to this many characters of table_html in the prompt.",
	)
	return p.parse_args()


def main() -> None:
	load_dotenv()

	args = parse_args()
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise ValueError("GEMINI_API_KEY environment variable is not set")

	client = genai.Client(api_key=api_key)

	items = load_dump(args.input)
	logger.info("Loaded %d table entries", len(items))

	results: List[Dict[str, Any]] = []
	for idx, item in enumerate(items, start=1):
		element_id = item.get("element_id")
		page_number = item.get("page_number")
		caption = item.get("caption")
		table_text = item.get("table_text")
		table_html = item.get("table_html")
		associated_text = item.get("associated_text") or []

		logger.info("Summarizing %d/%d: %s (page %s)", idx, len(items), element_id, page_number)
		try:
			summary = summarize_one(
				client,
				args.model,
				element_id=element_id,
				caption=caption,
				table_text=table_text,
				table_html=table_html,
				associated_text=associated_text,
				max_tokens=args.max_tokens,
				html_max_chars=args.html_max_chars,
			)
		except Exception as e:  # noqa: BLE001
			logger.error("GenAI call failed for %s: %s", element_id, e)
			summary = ""

		results.append(
			{
				"element_id": element_id,
				"page_number": page_number,
				"caption": caption,
				"summary": summary,
				"model": args.model,
				"has_table_text": bool(table_text and table_text.strip()),
				"has_html": bool(table_html and str(table_html).strip()),
			}
		)

	args.out.parent.mkdir(parents=True, exist_ok=True)
	with args.out.open("w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)
	logger.info("Wrote %d summaries to %s", len(results), args.out)


if __name__ == "__main__":
	main()

