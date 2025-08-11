"""
Normalize table records for ChromaDB ingestion by joining the LLM dump with table
summaries, producing JSONL records similar to papers_text.jsonl:

  id: element_id
  text: preferred text (summary if available, else table_text), prefixed with caption
  metadata: {
	doc_id, page_number, caption, table_html_file, source_path,
	element_type, has_summary, has_table_text, has_html
  }

Usage:
  uv run iterations/normalize_images.py \
	--dump ./files/extracted/tables/tables_llm_dump.json \
	--summaries ./files/extracted/tables/tables_llm_summaries.json \
	--out ./files/normalized/tables.jsonl

Note: Although named normalize_images.py, this script normalizes TABLES as per RAG plan.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("normalize_tables")


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def derive_doc_id(table_html_file: Optional[str]) -> Optional[str]:
	if not table_html_file:
		return None
	try:
		name = Path(table_html_file).name  # e.g., hydrocortisone_table_1.html
		stem = name.split(".")[0]
		base = stem.split("_table")[0]
		return base or None
	except Exception:  # noqa: BLE001
		return None


def normalize_record(item: Dict[str, Any], summary_by_id: Dict[str, Dict[str, Any]], source_path: str) -> Optional[Dict[str, Any]]:
	element_id = item.get("element_id")
	if not element_id:
		return None

	caption = item.get("caption")
	table_text = (item.get("table_text") or "").strip()
	table_html_file = item.get("table_html_file")
	page_number = item.get("page_number")
	associated_text = item.get("associated_text") or []

	sumrec = summary_by_id.get(element_id) or {}
	summary = (sumrec.get("summary") or "").strip()
	has_summary = bool(summary)
	has_table_text = bool(table_text)
	has_html = bool(item.get("table_html")) or bool(table_html_file)

	# Build embedding text: prefer summary; otherwise table_text. Always prefix with caption if present.
	parts: List[str] = []
	if caption:
		parts.append(f"Caption: {caption}")
	body = summary if has_summary else table_text
	if not body:
		# nothing to embed
		return None
	parts.append(body)

	text = "\n\n".join(parts).strip()

	doc_id = derive_doc_id(table_html_file) or "document"

	metadata = {
		"doc_id": doc_id,
		"page_number": page_number,
		"caption": caption,
		"table_html_file": table_html_file,
		"source_path": source_path,
		"element_type": "Table",
		"has_summary": has_summary,
		"has_table_text": has_table_text,
		"has_html": has_html,
	}

	return {"id": element_id, "text": text, "metadata": metadata}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Normalize tables by joining dump and summaries")
	p.add_argument("--dump", type=Path, default=Path("./files/extracted/tables/tables_llm_dump.json"))
	p.add_argument("--summaries", type=Path, default=Path("./files/extracted/tables/tables_llm_summaries.json"))
	p.add_argument("--out", type=Path, default=Path("./files/normalized/tables.jsonl"))
	p.add_argument("--min-chars", type=int, default=80, help="Skip rows with text shorter than this many characters")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	args.out.parent.mkdir(parents=True, exist_ok=True)

	dump = load_json(args.dump)
	if not isinstance(dump, list):
		raise ValueError("Dump must be a list")
	sums = load_json(args.summaries)
	if not isinstance(sums, list):
		raise ValueError("Summaries must be a list")

	summary_by_id = {r.get("element_id"): r for r in sums if r.get("element_id")}
	logger.info("Loaded %d tables and %d summaries", len(dump), len(summary_by_id))

	out_count = 0
	skipped = 0
	with args.out.open("w", encoding="utf-8") as f:
		for item in dump:
			rec = normalize_record(item, summary_by_id, source_path=str(args.dump))
			if not rec or len(rec["text"]) < args.min_chars:
				skipped += 1
				continue
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")
			out_count += 1

	logger.info("Wrote %d normalized table records to %s (skipped %d)", out_count, args.out, skipped)


if __name__ == "__main__":
	main()

