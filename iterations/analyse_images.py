"""
Analyse images from Unstructured elements JSON produced by v1_basic.py.

Outputs a JSON list where each item has:
- element_id
- caption (or "No caption")
- image_text (from element.text if present)
- image_base64 (from metadata or read from image_path if available)
- image_mime_type (if present)
- page_number (if present)

Usage options:
	# Preferred: point at the PDF, outputs will go to files/<stem>/extracted/images
	uv run iterations/analyse_images.py --pdf /path/to/your.pdf

	# Or use explicit input and out paths
	uv run iterations/analyse_images.py \
		--input ./files/<stem>/<stem>-output.json \
		--out ./files/<stem>/extracted/images/images_llm_dump.json
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("analyse_images")
from iterations.common_paths import (
	ensure_doc_dir,
	elements_json_path,
	images_dump_path,
)


def load_elements(path: Path) -> List[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("Expected a list of elements in the JSON file")
	return data


def build_caption_index(elements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
	"""Index captions by parent_id if available.

	Many FigureCaption elements carry metadata.parent_id referencing the image element_id.
	"""
	idx: Dict[str, List[str]] = {}
	for el in elements:
		if el.get("type") != "FigureCaption":
			continue
		meta = el.get("metadata") or {}
		parent_id = None
		if isinstance(meta, dict):
			parent_id = meta.get("parent_id") or meta.get("image_id")
		if not parent_id:
			continue
		txt = (el.get("text") or "").strip()
		if not txt:
			continue
		idx.setdefault(parent_id, []).append(txt)
	return idx


def find_nearby_caption(elements: List[Dict[str, Any]], image_index: int, window: int = 3) -> Optional[str]:
	"""Fallback: find a FigureCaption within the next few elements if no parent link exists."""
	n = len(elements)
	for j in range(image_index + 1, min(n, image_index + 1 + window)):
		el = elements[j]
		if el.get("type") == "FigureCaption":
			txt = (el.get("text") or "").strip()
			if txt:
				return txt
	return None


def read_file_as_base64(path: Path) -> Optional[str]:
	try:
		b = path.read_bytes()
		return base64.b64encode(b).decode("ascii")
	except Exception:
		return None


def extract_images_payload(
	elements: List[Dict[str, Any]],
	root: Path,
) -> List[Dict[str, Any]]:
	captions_by_parent = build_caption_index(elements)

	results: List[Dict[str, Any]] = []
	for i, el in enumerate(elements):
		if el.get("type") != "Image":
			continue

		meta = el.get("metadata") or {}
		text = (el.get("text") or "").strip() or None

		# Prefer metadata-provided caption fields
		caption: Optional[str] = None
		if isinstance(meta, dict):
			caption = (
				meta.get("image_caption")
				or meta.get("caption")
				or meta.get("figcaption")
				or None
			)
		# If not present, look up by parent_id mapping
		if not caption:
			caption_list = captions_by_parent.get(el.get("element_id", ""))
			if caption_list:
				caption = caption_list[0]
		# Last resort: nearby FigureCaption
		if not caption:
			caption = find_nearby_caption(elements, i)

		# Base64 image data
		image_b64: Optional[str] = None
		mime: Optional[str] = None
		page_number: Optional[int] = None
		if isinstance(meta, dict):
			image_b64 = meta.get("image_base64") or None
			mime = meta.get("image_mime_type") or None
			page_number = meta.get("page_number") if isinstance(meta.get("page_number"), int) else None
			# Try reading from image_path on disk if base64 missing
			if not image_b64:
				image_path_val = meta.get("image_path")
				if image_path_val:
					img_path = (root / str(image_path_val)).resolve() if not str(image_path_val).startswith("/") else Path(str(image_path_val))
					image_b64 = read_file_as_base64(img_path)

		results.append(
			{
				"element_id": el.get("element_id"),
				"caption": caption or "No caption",
				"image_text": text,
				"image_base64": image_b64,  # may be None if not available
				"image_mime_type": mime,
				"page_number": page_number,
			}
		)

	return results


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Create an LLM-ready dump of Image elements")
	p.add_argument("--pdf", type=Path, default=None, help="Optional: path to the source PDF to infer doc folder")
	p.add_argument("--input", type=Path, default=None, help="Optional: elements JSON path; inferred from --pdf if omitted")
	p.add_argument("--out", type=Path, default=None, help="Optional: output JSON path; inferred from --pdf if omitted")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	if args.pdf:
		doc_dir = ensure_doc_dir(args.pdf)
		input_path = elements_json_path(doc_dir)
		out_path = images_dump_path(doc_dir)
	else:
		if not args.input or not args.out:
			raise SystemExit("Either provide --pdf or both --input and --out")
		input_path = Path(args.input)
		out_path = Path(args.out)

	if not input_path.exists():
		logger.error("Input file not found: %s", input_path)
		raise SystemExit(1)

	logger.info("Loading elements from %s", input_path)
	elements = load_elements(input_path)

	root = Path(".").resolve()
	images = extract_images_payload(elements, root)
	logger.info("Found %d Image elements", len(images))

	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8") as f:
		json.dump(images, f, ensure_ascii=False, indent=2)
	logger.info("Wrote image dump to %s", out_path)


if __name__ == "__main__":
	main()

