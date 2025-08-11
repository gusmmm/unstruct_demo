"""
Path utilities to standardize RAG pipeline folder layout per document.

Layout for a PDF named <stem>.pdf:
  files/<stem>/
    <stem>.pdf
    <stem>-output.json
    <stem>-chunks-by-title.json
    <stem>-metadata-summary.json
    <stem>-references.json
    <stem>-references.txt
    extracted/
      images/
        <stem>_image_*.{jpg,png,...}
        images_llm_dump.json
        images_llm_summaries.json
      tables/
        <stem>_table_*.html
        tables_llm_dump.json
        tables_llm_summaries.json
    normalized/
      papers_text.jsonl
      images.jsonl
      tables.jsonl
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


def doc_stem_from_pdf(pdf_path: Path) -> str:
    return pdf_path.stem


def doc_dir_for_pdf(pdf_path: Path) -> Path:
    return Path("files") / doc_stem_from_pdf(pdf_path)


def ensure_doc_dir(pdf_path: Path, copy_pdf: bool = True) -> Path:
    doc_dir = doc_dir_for_pdf(pdf_path)
    doc_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = doc_dir / pdf_path.name
    if copy_pdf:
        try:
            if not target_pdf.exists() or target_pdf.stat().st_size != pdf_path.stat().st_size:
                shutil.copy2(pdf_path, target_pdf)
        except Exception:
            # Best-effort copy; ignore errors
            pass
    return doc_dir


def elements_json_path(doc_dir: Path, stem: Optional[str] = None) -> Path:
    s = stem or doc_dir.name
    return doc_dir / f"{s}-output.json"


def chunks_by_title_path(doc_dir: Path, stem: Optional[str] = None) -> Path:
    s = stem or doc_dir.name
    return doc_dir / f"{s}-chunks-by-title.json"


def metadata_summary_path(doc_dir: Path, stem: Optional[str] = None) -> Path:
    s = stem or doc_dir.name
    return doc_dir / f"{s}-metadata-summary.json"


def references_json_path(doc_dir: Path, stem: Optional[str] = None) -> Path:
    s = stem or doc_dir.name
    return doc_dir / f"{s}-references.json"


def references_txt_path(doc_dir: Path, stem: Optional[str] = None) -> Path:
    s = stem or doc_dir.name
    return doc_dir / f"{s}-references.txt"


def images_dir(doc_dir: Path) -> Path:
    return doc_dir / "extracted" / "images"


def images_dump_path(doc_dir: Path) -> Path:
    return images_dir(doc_dir) / "images_llm_dump.json"


def images_summaries_path(doc_dir: Path) -> Path:
    return images_dir(doc_dir) / "images_llm_summaries.json"


def tables_dir(doc_dir: Path) -> Path:
    return doc_dir / "extracted" / "tables"


def tables_dump_path(doc_dir: Path) -> Path:
    return tables_dir(doc_dir) / "tables_llm_dump.json"


def tables_summaries_path(doc_dir: Path) -> Path:
    return tables_dir(doc_dir) / "tables_llm_summaries.json"


def normalized_dir(doc_dir: Path) -> Path:
    return doc_dir / "normalized"


def normalized_papers_text_path(doc_dir: Path) -> Path:
    return normalized_dir(doc_dir) / "papers_text.jsonl"


def normalized_images_path(doc_dir: Path) -> Path:
    return normalized_dir(doc_dir) / "images.jsonl"


def normalized_tables_path(doc_dir: Path) -> Path:
    return normalized_dir(doc_dir) / "tables.jsonl"


def normalized_references_path(doc_dir: Path) -> Path:
    return normalized_dir(doc_dir) / "references.jsonl"
