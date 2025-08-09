"""
PDF processing pipeline using Unstructured OSS.

Features:
- Partition a PDF into elements (hi_res preferred for images/tables)
- Clean text elements
- Chunk text semantically by titles/headings
- Extract metadata summary (counts, languages, pages, title)
- Extract and save images and tables (images saved; tables as HTML)
- Extract bibliography/references section by heading

Usage (from repo root with uv):
  uv run iterations/v1_basic.py --pdf files/hydrocortisone.pdf --out-dir files

Notes:
- For image/table crops, hi_res strategy requires unstructured-inference (detectron2_onnx).
  If unavailable, we gracefully fall back to text-only extraction.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_to_json
from unstructured.cleaners.core import clean, replace_unicode_quotes


# -----------------------
# Logging configuration
# -----------------------
logger = logging.getLogger("pipeline")


# -----------------------
# Helpers
# -----------------------

TEXT_LIKE_CATEGORIES = {
	"Title",
	"Subtitle",
	"NarrativeText",
	"ListItem",
	"Header",
	"Footer",
	"UncategorizedText",
	"FigureCaption",
}


def ensure_dir(path: Path) -> None:
	"""Create directory if it doesn't exist."""
	path.mkdir(parents=True, exist_ok=True)


def is_text_like(category: Optional[str]) -> bool:
	return bool(category and category in TEXT_LIKE_CATEGORIES)


def _element_category(el: Any) -> str:
	# Unstructured elements have .category in recent versions; fallback to class name
	try:
		return getattr(el, "category") or el.__class__.__name__
	except Exception:
		return el.__class__.__name__


def _element_text(el: Any) -> str:
	return getattr(el, "text", "") or ""


def _element_metadata(el: Any) -> Any:
	return getattr(el, "metadata", None)


def partition_document(
	pdf_path: Path,
	include_page_breaks: bool = True,
	try_extract_image_blocks: bool = True,
	ocr_languages: Optional[List[str]] = None,
) -> List[Any]:
	"""Partition the PDF into Unstructured elements.

	Attempts hi_res with image/table crops; falls back to auto if unavailable.
	"""
	logger.info("Partitioning PDF: %s", pdf_path)

	elements: List[Any] = []

	if try_extract_image_blocks:
		try:
			logger.debug("Trying hi_res strategy with image block extraction")
			elements = partition_pdf(
				filename=str(pdf_path),
				include_page_breaks=include_page_breaks,
				strategy="hi_res",
				languages=ocr_languages,
				# Newer, preferred knobs for crops:
				extract_image_block_types=["Image", "Table"],
				extract_image_block_to_payload=True,
			)
			logger.info("Partitioned with hi_res and image blocks: %d elements", len(elements))
			return elements
		except Exception as e:  # noqa: BLE001
			logger.warning(
				"hi_res with image extraction failed (%s). Falling back to auto without crops.",
				e,
			)

	# Fallback: auto strategy (text-first). May still produce Table elements, but no crops.
	elements = partition_pdf(
		filename=str(pdf_path),
		include_page_breaks=include_page_breaks,
		strategy="auto",
		languages=ocr_languages,
	)
	logger.info("Partitioned with auto strategy: %d elements", len(elements))
	return elements


def clean_text_elements(elements: List[Any]) -> List[Any]:
	"""Apply conservative text cleaning to text-like elements in-place."""
	logger.info("Cleaning text elements")
	for el in elements:
		category = _element_category(el)
		if is_text_like(category):
			try:
				# Normalize unicode quotes, bullets/whitespace/dashes
				el.apply(replace_unicode_quotes)
				el.apply(
					lambda t: clean(
						t,
						bullets=True,
						extra_whitespace=True,
						dashes=True,
						trailing_punctuation=False,
						lowercase=False,
					)
				)
			except Exception as e:  # noqa: BLE001
				logger.debug("Skipping cleaning for element due to error: %s", e)
	return elements


def chunk_semantic_by_title(
	elements: List[Any],
	max_characters: int = 1200,
	new_after_n_chars: int = 900,
	overlap: int = 100,
	multipage_sections: bool = True,
) -> List[Any]:
	"""Chunk elements by titles/headings preserving section boundaries."""
	logger.info(
		"Chunking by title (max=%d, soft=%d, overlap=%d, multipage=%s)",
		max_characters,
		new_after_n_chars,
		overlap,
		multipage_sections,
	)
	chunks = chunk_by_title(
		elements,
		max_characters=max_characters,
		new_after_n_chars=new_after_n_chars,
		overlap=overlap,
		multipage_sections=multipage_sections,
	)
	logger.info("Created %d chunks", len(chunks))
	return chunks


def extract_images(
	elements: List[Any],
	output_dir: Path,
	stem: str,
) -> List[Path]:
	"""Extract Base64-embedded image/table crops (requires hi_res with image blocks)."""
	ensure_dir(output_dir)
	saved: List[Path] = []

	def pick_ext(mime: Optional[str]) -> str:
		return {
			"image/jpeg": ".jpg",
			"image/png": ".png",
			"image/webp": ".webp",
			"image/tiff": ".tiff",
			"image/bmp": ".bmp",
		}.get(mime or "", ".jpg")

	idx = 0
	for el in elements:
		meta = _element_metadata(el)
		if not meta:
			continue
		md = getattr(meta, "to_dict", lambda: {})()
		if "image_base64" in md:
			mime = md.get("image_mime_type")
			ext = pick_ext(mime)
			idx += 1
			out_path = output_dir / f"{stem}_image_{idx}{ext}"
			try:
				data = base64.b64decode(md["image_base64"])  # type: ignore[index]
				out_path.write_bytes(data)
				saved.append(out_path)
			except Exception as e:  # noqa: BLE001
				logger.warning("Failed to write image %s: %s", out_path, e)
	logger.info("Saved %d images (image/table crops)", len(saved))
	return saved


def extract_tables(
	elements: List[Any],
	output_dir: Path,
	stem: str,
) -> List[Path]:
	"""Save Table elements' HTML representation to disk."""
	ensure_dir(output_dir)
	saved: List[Path] = []
	t_idx = 0
	for el in elements:
		if _element_category(el) == "Table":
			meta = _element_metadata(el)
			if not meta:
				continue
			md = getattr(meta, "to_dict", lambda: {})()
			html = md.get("text_as_html")
			if not html:
				# As a fallback, write text content
				html = f"<pre>{_element_text(el)}</pre>"
			t_idx += 1
			out_path = output_dir / f"{stem}_table_{t_idx}.html"
			try:
				out_path.write_text(str(html))
				saved.append(out_path)
			except Exception as e:  # noqa: BLE001
				logger.warning("Failed to write table %s: %s", out_path, e)
	logger.info("Saved %d tables (HTML)", len(saved))
	return saved


def build_metadata_summary(elements: List[Any]) -> Dict[str, Any]:
	"""Compute a simple document-level metadata summary from element metadata."""
	counts: Dict[str, int] = {}
	languages: List[str] = []
	max_page = 0
	first_title: Optional[str] = None

	for el in elements:
		cat = _element_category(el)
		counts[cat] = counts.get(cat, 0) + 1

		meta = _element_metadata(el)
		if meta:
			md = getattr(meta, "to_dict", lambda: {})()
			# accumulate languages
			langs = md.get("languages") or []
			for lg in langs:
				if lg not in languages:
					languages.append(lg)
			# page number
			pn = md.get("page_number")
			if isinstance(pn, int) and pn > max_page:
				max_page = pn

		if not first_title and cat == "Title":
			first_title = _element_text(el).strip() or None

	return {
		"element_counts": counts,
		"languages": languages,
		"num_pages": max_page or None,
		"document_title": first_title,
	}


def extract_references_section(elements: List[Any]) -> Dict[str, Any]:
	"""Extract a bibliography/references section by locating a heading and capturing following text.

	Heuristics:
	- Find Title whose text matches common reference section names.
	- Collect subsequent text-like elements until the next Title or end.
	"""
	logger.info("Extracting references/bibliography section")
	target_headers = {"references", "bibliography", "works cited"}

	start_idx: Optional[int] = None
	for i, el in enumerate(elements):
		if _element_category(el) == "Title":
			txt = _element_text(el).strip().lower()
			# Normalize shorts: often just "References"
			if any(h in txt for h in target_headers):
				start_idx = i
				break

	if start_idx is None:
		logger.info("No references section found")
		return {"found": False, "heading": None, "content": "", "items": []}

	# Capture until next Title
	heading = _element_text(elements[start_idx]).strip()
	collected: List[str] = []
	items: List[str] = []
	for el in elements[start_idx + 1 :]:
		if _element_category(el) == "Title":
			break
		if is_text_like(_element_category(el)):
			txt = _element_text(el).strip()
			if not txt:
				continue
			collected.append(txt)
			# Roughly split individual references by newline or bullety patterns is tricky;
			# keep both aggregate and per-element items for downstream use.
			items.append(txt)

	return {
		"found": True,
		"heading": heading,
		"content": "\n".join(collected).strip(),
		"items": items,
	}


def save_json(obj: Any, path: Path) -> None:
	ensure_dir(path.parent)
	with path.open("w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, ensure_ascii=False)


def run_pipeline(pdf_path: Path, out_dir: Path) -> Dict[str, Any]:
	"""Run the complete pipeline and write artifacts under out_dir.

	Returns basic summary of outputs.
	"""
	stem = pdf_path.stem

	# 1) Partition
	elements = partition_document(pdf_path)

	# 2) Clean
	elements = clean_text_elements(elements)

	# 3) Save raw elements JSON
	raw_json_path = out_dir / f"{stem}-output.json"
	ensure_dir(raw_json_path.parent)
	elements_to_json(elements=elements, filename=str(raw_json_path))

	# 4) Chunk by title
	chunks = chunk_semantic_by_title(elements)
	chunks_json_path = out_dir / f"{stem}-chunks-by-title.json"
	elements_to_json(elements=chunks, filename=str(chunks_json_path))

	# 5) Extract metadata summary
	summary = build_metadata_summary(elements)
	summary_json_path = out_dir / f"{stem}-metadata-summary.json"
	save_json(summary, summary_json_path)

	# 6) Extract images and tables
	images_dir = out_dir / "extracted" / "images"
	tables_dir = out_dir / "extracted" / "tables"
	image_files = extract_images(elements, images_dir, stem=stem)
	table_files = extract_tables(elements, tables_dir, stem=stem)

	# 7) Extract references
	refs = extract_references_section(elements)
	refs_json_path = out_dir / f"{stem}-references.json"
	save_json(refs, refs_json_path)
	# Additionally, a simple .txt for quick viewing
	if refs.get("found") and refs.get("content"):
		(out_dir / f"{stem}-references.txt").write_text(refs["content"], encoding="utf-8")

	return {
		"raw_elements_json": str(raw_json_path),
		"chunks_json": str(chunks_json_path),
		"metadata_summary_json": str(summary_json_path),
		"references_json": str(refs_json_path),
		"images_saved": [str(p) for p in image_files],
		"tables_saved": [str(p) for p in table_files],
	}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Unstructured PDF pipeline")
	parser.add_argument(
		"--pdf",
		required=True,
		help="Path to input PDF",
	)
	parser.add_argument(
		"--out-dir",
		default="files",
		help="Directory to write outputs (default: files)",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="Logging level",
	)
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
	args = parse_args(argv)

	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)

	pdf_path = Path(args.pdf).expanduser().resolve()
	out_dir = Path(args.out_dir).expanduser().resolve()

	if not pdf_path.exists():
		logger.error("PDF not found: %s", pdf_path)
		raise SystemExit(1)

	logger.info("Running pipeline for %s", pdf_path)
	outputs = run_pipeline(pdf_path, out_dir)

	# Compact summary to log
	logger.info("Artifacts written:")
	for k, v in outputs.items():
		if isinstance(v, list):
			logger.info("  %s: %d items", k, len(v))
		else:
			logger.info("  %s: %s", k, v)


if __name__ == "__main__":
	main()

