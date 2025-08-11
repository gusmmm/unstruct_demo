"""
Normalize image (figure) records for ChromaDB ingestion by joining the LLM dump
with image summaries, producing JSONL records similar to papers_text.jsonl:

  id: element_id
  text: preferred text (summary if available; else caption + on-image text)
  metadata: {
	doc_id, page_number, caption, source_path,
	element_type, has_summary, has_image_binary, image_mime_type
  }

Usage:
  uv run iterations/normalize_images.py \
	--dump ./files/extracted/images/images_llm_dump.json \
	--summaries ./files/extracted/images/images_llm_summaries.json \
	--out ./files/normalized/images.jsonl \
	--doc-id hydrocortisone
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("normalize_images")


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def build_text(caption: Optional[str], summary: Optional[str], on_image_text: Optional[str]) -> Optional[str]:
	caption = (caption or "").strip()
	summary = (summary or "").strip()
	on_image_text = (on_image_text or "").strip()

	parts: List[str] = []
	if caption:
		parts.append(f"Caption: {caption}")
	if summary:
		parts.append(summary)
	else:
		# Fallback: no summary, use on-image text if available
		if on_image_text:
			parts.append(f"On-image text: {on_image_text}")

	text = "\n\n".join(parts).strip()
	return text or None


def normalize_record(item: Dict[str, Any], summary_by_id: Dict[str, Dict[str, Any]], source_path: str, default_doc_id: str) -> Optional[Dict[str, Any]]:
	element_id = item.get("element_id")
	if not element_id:
		return None
	caption = item.get("caption")
	on_image_text = item.get("image_text")
	image_b64 = item.get("image_base64")
	image_mime = item.get("image_mime_type")
	page_number = item.get("page_number")

	sumrec = summary_by_id.get(element_id) or {}
	summary = sumrec.get("summary")

	text = build_text(caption, summary, on_image_text)
	if not text:
		return None

	metadata = {
		"doc_id": default_doc_id,
		"page_number": page_number,
		"caption": caption,
		"source_path": source_path,
		"element_type": "Image",
		"has_summary": bool((summary or "").strip()),
		"has_image_binary": bool(image_b64),
		"image_mime_type": image_mime,
	}

	return {"id": element_id, "text": text, "metadata": metadata}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Normalize images by joining dump and summaries")
	p.add_argument("--dump", type=Path, default=Path("./files/extracted/images/images_llm_dump.json"))
	p.add_argument("--summaries", type=Path, default=Path("./files/extracted/images/images_llm_summaries.json"))
	p.add_argument("--out", type=Path, default=Path("./files/normalized/images.jsonl"))
	p.add_argument("--doc-id", type=str, default="document", help="Doc ID to store in metadata (e.g., hydrocortisone)")
	p.add_argument("--min-chars", type=int, default=60, help="Skip rows with text shorter than this many characters")
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
	logger.info("Loaded %d images and %d summaries", len(dump), len(summary_by_id))

	out_count = 0
	skipped = 0
	with args.out.open("w", encoding="utf-8") as f:
		for item in dump:
			rec = normalize_record(item, summary_by_id, source_path=str(args.dump), default_doc_id=args.doc_id)
			if not rec or len(rec["text"]) < args.min_chars:
				skipped += 1
				continue
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")
			out_count += 1

	logger.info("Wrote %d normalized image records to %s (skipped %d)", out_count, args.out, skipped)


if __name__ == "__main__":
	main()

