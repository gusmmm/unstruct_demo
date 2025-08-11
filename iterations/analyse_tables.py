"""
Extract all Table elements from an Unstructured elements JSON and build a
well-organized JSON dump for RAG.

For each table, we include:
- element_id
- page_number (if present)
- caption (from metadata or linked/nearby captions; default "No caption")
- table_text (raw text on the element)
- table_html (metadata.text_as_html if available)
- associated_text (nearby NarrativeText / ListItem / UncategorizedText)

Usage options:
    # Preferred: point at the PDF, outputs will go to files/<stem>/extracted/tables
    uv run iterations/analyse_tables.py --pdf /path/to/your.pdf

    # Or explicit paths
    uv run iterations/analyse_tables.py \
        --input ./files/<stem>/<stem>-output.json \
        --out ./files/<stem>/extracted/tables/tables_llm_dump.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("analyse_tables")
from .common_paths import (
    ensure_doc_dir,
    elements_json_path,
    tables_dump_path,
    tables_dir as tables_html_dir,
)


TEXT_LIKE = {"NarrativeText", "ListItem", "UncategorizedText"}
CAPTION_TYPES = {"FigureCaption", "TableCaption", "Caption"}


def load_elements(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of elements in the JSON file")
    return data


def build_caption_index(elements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Index captions by parent_id if available.

    Some caption elements carry metadata.parent_id referencing the table element_id.
    """
    idx: Dict[str, List[str]] = {}
    for el in elements:
        if el.get("type") not in CAPTION_TYPES:
            continue
        meta = el.get("metadata") or {}
        parent_id = None
        if isinstance(meta, dict):
            parent_id = meta.get("parent_id") or meta.get("table_id")
        if not parent_id:
            continue
        txt = (el.get("text") or "").strip()
        if not txt:
            continue
        idx.setdefault(parent_id, []).append(txt)
    return idx


def find_nearby_caption(elements: List[Dict[str, Any]], table_index: int, window: int = 3) -> Optional[str]:
    """Fallback: find a caption-like element near the table if no explicit parent link exists."""
    n = len(elements)
    # Look ahead a few elements
    for j in range(table_index + 1, min(n, table_index + 1 + window)):
        el = elements[j]
        if el.get("type") in CAPTION_TYPES:
            txt = (el.get("text") or "").strip()
            if txt:
                return txt
        # Occasionally tables have a Title like "Table 1" immediately after/before
        if el.get("type") == "Title":
            t = (el.get("text") or "").strip()
            if t.lower().startswith("table"):
                return t
    # Look backward a few elements
    for j in range(max(0, table_index - window), table_index):
        el = elements[j]
        if el.get("type") in CAPTION_TYPES:
            txt = (el.get("text") or "").strip()
            if txt:
                return txt
        if el.get("type") == "Title":
            t = (el.get("text") or "").strip()
            if t.lower().startswith("table"):
                return t
    return None


def collect_associated_text(elements: List[Dict[str, Any]], table_index: int, window: int = 4) -> List[str]:
    lines: List[str] = []
    n = len(elements)
    # Backward
    for j in range(max(0, table_index - window), table_index):
        el = elements[j]
        if el.get("type") in TEXT_LIKE:
            t = (el.get("text") or "").strip()
            if t:
                lines.append(t)
    # Forward
    for j in range(table_index + 1, min(n, table_index + 1 + window)):
        el = elements[j]
        if el.get("type") in TEXT_LIKE:
            t = (el.get("text") or "").strip()
            if t:
                lines.append(t)
        # Stop early if we hit a new section/table title to avoid bleeding too far
        if el.get("type") == "Title":
            break
        if el.get("type") == "Table":
            break
    return lines


def extract_tables(
    elements: List[Dict[str, Any]],
    html_files: Optional[List[Path]] = None,
    html_max_chars: int = 200_000,
) -> List[Dict[str, Any]]:
    captions_by_parent = build_caption_index(elements)

    results: List[Dict[str, Any]] = []
    table_index_in_doc = -1  # monotonic counter of tables as encountered
    for i, el in enumerate(elements):
        if el.get("type") != "Table":
            continue
        table_index_in_doc += 1

        meta = el.get("metadata") or {}
        page_number = meta.get("page_number") if isinstance(meta, dict) else None
        caption = None
        if isinstance(meta, dict):
            caption = meta.get("table_caption") or meta.get("caption") or meta.get("figcaption")

        if not caption:
            # parent-link captions
            pid = el.get("element_id", "")
            if pid and pid in captions_by_parent:
                caption = captions_by_parent[pid][0]
        if not caption:
            caption = find_nearby_caption(elements, i) or "No caption"

        table_text = (el.get("text") or "").strip() or None
        table_html = None
        if isinstance(meta, dict):
            table_html = meta.get("text_as_html")

        # If no HTML in metadata, try to attach from external HTML files by order
        html_file_path: Optional[Path] = None
        if not table_html and html_files:
            if 0 <= table_index_in_doc < len(html_files):
                html_file_path = html_files[table_index_in_doc]
                try:
                    table_html = html_file_path.read_text(encoding="utf-8")
                    if html_max_chars and len(table_html) > html_max_chars:
                        table_html = table_html[:html_max_chars]
                except Exception:
                    table_html = None

        assoc = collect_associated_text(elements, i)

        results.append(
            {
                "element_id": el.get("element_id"),
                "page_number": page_number,
                "caption": caption or "No caption",
                "table_text": table_text,
                "table_html": table_html,
                "table_html_file": str(html_file_path) if html_file_path else None,
                "associated_text": assoc,
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Table elements to a JSON dump for RAG")
    p.add_argument("--pdf", type=Path, default=None, help="Optional: path to the source PDF to infer doc folder")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--html-dir",
        type=Path,
        default=None,
        help="Optional directory containing extracted table HTML files (matched by order)",
    )
    p.add_argument(
        "--html-max-chars",
        type=int,
        default=200_000,
        help="Max characters of HTML to include per table to avoid huge dumps (0 to disable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.pdf:
        doc_dir = ensure_doc_dir(args.pdf)
        input_path = elements_json_path(doc_dir)
        out_path = tables_dump_path(doc_dir)
        default_html_dir = tables_html_dir(doc_dir)
        if args.html_dir is None:
            args.html_dir = default_html_dir
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

    html_files: Optional[List[Path]] = None
    if args.html_dir and args.html_dir.exists():
        # Collect HTML files sorted to align with table order
        html_files = sorted(args.html_dir.glob("*.html"))
        logger.info("Found %d HTML files in %s", len(html_files), args.html_dir)

    tables = extract_tables(elements, html_files=html_files, html_max_chars=args.html_max_chars)
    logger.info("Selected %d Table elements", len(tables))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    logger.info("Wrote tables dump to %s", out_path)


if __name__ == "__main__":
    main()
