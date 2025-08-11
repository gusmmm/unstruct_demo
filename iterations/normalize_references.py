"""
Normalize references into a JSONL file for Chroma ingestion.

Input sources (inferred via --pdf):
  - files/<stem>/<stem>-references.json (from v1_basic.py)
  - files/<stem>/<stem>-references.txt (optional)

Output:
  - files/<stem>/normalized/references.jsonl with one record per line:
    { id, text, metadata: { doc_id, source_path, heading_found } }
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from iterations.common_paths import (
    ensure_doc_dir,
    normalized_references_path,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("normalize_references")


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize references for indexing")
    p.add_argument("--pdf", type=Path, required=True, help="Path to the source PDF to infer doc folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    doc_dir = ensure_doc_dir(args.pdf)
    stem = doc_dir.name

    refs_json = doc_dir / f"{stem}-references.json"
    refs_txt = doc_dir / f"{stem}-references.txt"
    out_path = normalized_references_path(doc_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(refs_json) or {}
    found = bool(data.get("found"))
    items: List[str] = []
    if isinstance(data.get("items"), list):
        items = [str(x).strip() for x in data["items"] if str(x).strip()]

    # If TXT exists and has content, use it to supplement/replace items when JSON list is empty
    if refs_txt.exists() and not items:
        try:
            txt = refs_txt.read_text(encoding="utf-8")
            # naive split per line; downstream can re-parse as needed
            for line in txt.splitlines():
                line = line.strip()
                if line:
                    items.append(line)
        except Exception:
            pass

    # Build JSONL rows
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, citation in enumerate(items, start=1):
            cid = f"ref_{i}"
            rec = {
                "id": cid,
                "text": citation,
                "metadata": {
                    "doc_id": stem,
                    "source_path": str(refs_json if refs_json.exists() else refs_txt),
                    "heading_found": found,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d reference records to %s", written, out_path)


if __name__ == "__main__":
    main()
