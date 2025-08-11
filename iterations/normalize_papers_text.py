"""
Extract narrative text chunks from Unstructured elements JSON, grouped by section titles,
preserving semantic structure for RAG. Outputs JSONL suitable for a `papers_text` collection:

  id: element_id (prefer the Title element id for the chunk, else first element id, else synthetic)
  text: chunk text (concatenated narrative/list items under the section title)
  metadata: {
	doc_id, page_number, title, section, source_path, element_type, parent_id,
	first_page, last_page
  }

Usage:
  uv run iterations/papers_text.py \
	--input ./files/hydrocortisone-output.json \
	--out ./files/normalized/papers_text.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("papers_text")


TEXT_TYPES = {"NarrativeText", "ListItem"}
TITLE_TYPE = "Title"
SKIP_TYPES = {"Header", "Footer", "PageBreak", "FigureCaption", "Table"}


def load_elements(path: Path) -> List[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("Input JSON must be a list of elements")
	return data


def get_text(el: Dict[str, Any]) -> str:
	t = el.get("text")
	if isinstance(t, str):
		return t.strip()
	# Some exports might nest text under "metadata" or "data"
	md = el.get("metadata") or {}
	if isinstance(md, dict):
		mt = md.get("text")
		if isinstance(mt, str):
			return mt.strip()
	return ""


def get_page(el: Dict[str, Any]) -> Optional[int]:
	md = el.get("metadata") or {}
	if isinstance(md, dict):
		pn = md.get("page_number")
		try:
			return int(pn) if pn is not None else None
		except Exception:  # noqa: BLE001
			return None
	return None


def get_parent_id(el: Dict[str, Any]) -> Optional[str]:
	md = el.get("metadata") or {}
	if isinstance(md, dict):
		pid = md.get("parent_id")
		if isinstance(pid, str):
			return pid
	return None


def canonical_section(title_text: str) -> str:
	tt = (title_text or "").strip().lower()
	# common scientific sections
	mapping = [
		("abstract", "abstract"),
		("background", "background"),
		("introduction", "introduction"),
		("methods", "methods"),
		("method", "methods"),
		("materials and methods", "methods"),
		("results", "results"),
		("discussion", "discussion"),
		("conclusion", "conclusion"),
		("conclusions", "conclusion"),
		("supplementary", "supplementary"),
		("appendix", "appendix"),
		("references", "references"),
		("acknowledgment", "acknowledgments"),
		("acknowledgement", "acknowledgments"),
		("funding", "funding"),
		("ethics", "ethics"),
		("trial registration", "trial_registration"),
		("limitations", "limitations"),
	]
	for prefix, canon in mapping:
		if tt.startswith(prefix):
			return canon
	# avoid treating figure/table titles as sections
	if tt.startswith("table ") or tt.startswith("figure "):
		return "nonsection_title"
	return "section"


def iter_section_blocks(elements: List[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
	"""Yield (title_el, content_els) for each section, where title_el is a Title element.
	Content includes contiguous NarrativeText/ListItem until the next Title.
	Titles that look like figure/table headings are yielded with empty content.
	"""
	current_title: Optional[Dict[str, Any]] = None
	buffer: List[Dict[str, Any]] = []

	def flush():
		nonlocal current_title, buffer
		if current_title is not None:
			yield (current_title, buffer)
		current_title, buffer = None, []

	i = 0
	n = len(elements)
	while i < n:
		el = elements[i]
		etype = el.get("type")
		if etype == TITLE_TYPE:
			# start a new block
			if current_title is not None:
				# flush previous block
				yield from flush()
			current_title = el
			buffer = []
			i += 1
			continue

		if etype in TEXT_TYPES:
			if current_title is not None:
				buffer.append(el)
			# else we are before the first title; ignore or collect under a synthetic preamble?
		# ignore skip types and others by default
		i += 1

	if current_title is not None:
		yield from flush()


def build_chunk(title_el: Dict[str, Any], content_els: List[Dict[str, Any]], doc_id: str, source_path: str) -> Optional[Dict[str, Any]]:
	title_text = get_text(title_el)
	sect = canonical_section(title_text)

	# Avoid non-section titles like Table/Figure headings
	if sect == "nonsection_title":
		return None

	parts: List[str] = []
	for el in content_els:
		t = get_text(el)
		if not t:
			continue
		if el.get("type") == "ListItem":
			parts.append(f"- {t}")
		else:
			parts.append(t)

	text = "\n\n".join(parts).strip()
	if not text:
		# empty section; skip
		return None

	page_numbers = [p for p in (get_page(title_el), *(get_page(e) for e in content_els)) if p is not None]
	first_page = min(page_numbers) if page_numbers else None
	last_page = max(page_numbers) if page_numbers else None

	chunk_id = title_el.get("element_id") or (content_els[0].get("element_id") if content_els else None)
	if not chunk_id:
		# generate synthetic id
		chunk_id = f"chunk_{abs(hash(title_text))}_{first_page or 0}"

	metadata = {
		"doc_id": doc_id,
		"page_number": first_page,
		"title": title_text,
		"section": sect,
		"source_path": source_path,
		"element_type": "SectionChunk",
		"parent_id": get_parent_id(title_el),
		"first_page": first_page,
		"last_page": last_page,
	}

	record = {"id": chunk_id, "text": text, "metadata": metadata}
	return record


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Build papers_text chunks grouped by section titles")
	p.add_argument("--input", type=Path, default=Path("./files/hydrocortisone-output.json"))
	p.add_argument("--out", type=Path, default=Path("./files/normalized/papers_text.jsonl"))
	p.add_argument("--min-chars", type=int, default=200, help="Skip chunks with text shorter than this many characters")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	args.out.parent.mkdir(parents=True, exist_ok=True)

	elements = load_elements(args.input)
	logger.info("Loaded %d elements", len(elements))

	# Derive doc_id from input filename (e.g., hydrocortisone-output.json -> hydrocortisone)
	stem = args.input.stem
	if stem.endswith("-output"):
		doc_id = stem[:-7]
	else:
		doc_id = stem

	records: List[Dict[str, Any]] = []
	for title_el, content_els in iter_section_blocks(elements):
		rec = build_chunk(title_el, content_els, doc_id=doc_id, source_path=str(args.input))
		if not rec:
			continue
		if len(rec["text"]) < args.min_chars:
			continue
		records.append(rec)

	with args.out.open("w", encoding="utf-8") as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")
	logger.info("Wrote %d section chunks to %s", len(records), args.out)


if __name__ == "__main__":
	main()

