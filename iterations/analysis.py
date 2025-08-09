"""
Generate a static, visually-attractive HTML file that analyzes and renders the
contents of the Unstructured output JSON (files/hydrocortisone-output.json),
including text, images, tables, and references.

Usage (from project root):
	uv run python -m iterations.analysis

This will produce: files/hydrocortisone-analysis.html

Notes:
- Only standard library is used. No network calls.
- Images and tables are pulled from files/extracted/{images,tables} if present.
- Logs are written to stdout using logging (per project guidelines).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ------------- Data models


@dataclass
class Element:
	type: str
	text: str = ""
	metadata: Dict[str, Any] | None = None
	element_id: str | None = None
	parent_id: str | None = None


# ------------- Core rendering logic


class HydrocortisoneHtmlReport:
	"""Builds a static HTML report from Unstructured JSON and extracted assets."""

	def __init__(self, project_root: Path | None = None) -> None:
		self.project_root = project_root or Path(__file__).resolve().parent.parent
		self.files_dir = self.project_root / "files"
		self.input_json = self.files_dir / "hydrocortisone-output.json"
		self.tables_dir = self.files_dir / "extracted" / "tables"
		self.images_dir = self.files_dir / "extracted" / "images"
		self.references_txt = self.files_dir / "hydrocortisone-references.txt"
		self.references_json = self.files_dir / "hydrocortisone-references.json"
		self.output_html = self.files_dir / "hydrocortisone-analysis.html"

		logger.debug("Project root: %s", self.project_root)

	# ---------- IO helpers

	def _load_elements(self) -> List[Element]:
		if not self.input_json.exists():
			raise FileNotFoundError(f"Input JSON not found: {self.input_json}")
		logger.info("Reading JSON: %s", self.input_json)
		raw = json.loads(self.input_json.read_text(encoding="utf-8"))
		elements: List[Element] = []
		for obj in raw:
			etype = obj.get("type") or ""
			text = obj.get("text") or ""
			meta = obj.get("metadata") or {}
			parent_id = None
			if isinstance(meta, dict):
				parent_id = meta.get("parent_id") or obj.get("metadata", {}).get("parent_id") or obj.get("parent_id")
			elements.append(
				Element(
					type=str(etype),
					text=str(text),
					metadata=meta if isinstance(meta, dict) else {},
					element_id=obj.get("element_id"),
					parent_id=parent_id,
				)
			)
		logger.info("Loaded %d elements", len(elements))
		return elements

	def _load_images(self) -> List[Path]:
		images: List[Path] = []
		if self.images_dir.exists():
			images = sorted(self.images_dir.glob("hydrocortisone_image_*.jpg"), key=_numeric_sort_key)
		logger.info("Found %d images", len(images))
		return images

	def _load_tables(self) -> List[Tuple[Path, str]]:
		tables: List[Tuple[Path, str]] = []
		if self.tables_dir.exists():
			for p in sorted(self.tables_dir.glob("hydrocortisone_table_*.html"), key=_numeric_sort_key):
				try:
					html = p.read_text(encoding="utf-8")
				except UnicodeDecodeError:
					html = p.read_text(encoding="latin-1")
				tables.append((p, html))
		logger.info("Found %d tables", len(tables))
		return tables

	def _load_references(self) -> List[str]:
		# Prefer the txt for readability; fallback to JSON list
		refs: List[str] = []
		if self.references_txt.exists():
			txt = self.references_txt.read_text(encoding="utf-8", errors="ignore").strip()
			# Split on blank lines or numbered prefixes
			parts = [r.strip() for r in txt.splitlines() if r.strip()]
			refs = parts
		elif self.references_json.exists():
			try:
				data = json.loads(self.references_json.read_text(encoding="utf-8"))
				if isinstance(data, list):
					refs = [str(x) for x in data]
			except Exception as e:  # pragma: no cover - defensive
				logger.warning("Failed to parse references JSON: %s", e)
		logger.info("Loaded %d references", len(refs))
		return refs

	# ---------- Analysis helpers

	def _summarize(self, elements: List[Element]) -> Dict[str, Any]:
		type_counts = Counter(e.type for e in elements if e.type)
		pages = [safe_int(e.metadata.get("page_number")) for e in elements if isinstance(e.metadata, dict) and e.metadata.get("page_number") is not None]
		unique_pages = sorted(set(p for p in pages if p is not None))
		empty_text = sum(1 for e in elements if not (e.text or "").strip())
		sample_titles = [e.text.strip() for e in elements if e.type == "Title" and (e.text or "").strip()][:5]
		return {
			"total_elements": len(elements),
			"type_counts": dict(type_counts),
			"page_count": len(unique_pages) or None,
			"pages": unique_pages,
			"empty_text": empty_text,
			"sample_titles": sample_titles,
		}

	# ---------- Renderers

	def build(self) -> Path:
		elements = self._load_elements()
		images = self._load_images()
		tables = self._load_tables()
		references = self._load_references()
		summary = self._summarize(elements)

		logger.info("Rendering HTML -> %s", self.output_html)
		html = self._render_full_html(elements, images, tables, references, summary)
		self.output_html.write_text(html, encoding="utf-8")
		logger.info("Done: %s", self.output_html)
		return self.output_html

	def _render_full_html(
		self,
		elements: List[Element],
		images: List[Path],
		tables: List[Tuple[Path, str]],
		references: List[str],
		summary: Dict[str, Any],
	) -> str:
		# Compute relative paths from output_html to assets under files/
		# output_html sits in files/, so we want paths relative to it.
		rel_images = [p.relative_to(self.files_dir) if p.is_absolute() else p for p in images]

		body_sections: List[str] = []

		# Top banner
		body_sections.append(
			f"""
			<header class="banner">
			  <h1>Hydrocortisone Paper — Unstructured JSON Explorer</h1>
			  <p class="subtitle">Generated on {escape(datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>
			  <nav class="nav">
				<a href="#summary">Summary</a>
				<a href="#document">Document</a>
				<a href="#images">Images</a>
				<a href="#tables">Tables</a>
				<a href="#references">References</a>
			  </nav>
			</header>
			"""
		)

		# Summary section
		body_sections.append(self._render_summary_section(summary))

		# Main document stream
		body_sections.append("<section id=\"document\" class=\"section\"><h2>Document</h2>")
		body_sections.append(self._render_document_stream(elements, rel_images, tables))
		body_sections.append("</section>")

		# Image gallery (all images)
		body_sections.append("<section id=\"images\" class=\"section\"><h2>Images</h2>")
		if rel_images:
			gallery = [
				f"<figure class=\"card img\"><img loading=\"lazy\" src=\"{escape(str(p))}\" alt=\"Figure {i+1}\"><figcaption>Figure {i+1}</figcaption></figure>"
				for i, p in enumerate(rel_images)
			]
			body_sections.append(f"<div class=\"grid\">{''.join(gallery)}</div>")
		else:
			body_sections.append("<p class=\"muted\">No images found.</p>")
		body_sections.append("</section>")

		# Tables section (all tables)
		body_sections.append("<section id=\"tables\" class=\"section\"><h2>Tables</h2>")
		if tables:
			rendered = []
			for i, (path, html) in enumerate(tables, start=1):
				rendered.append(
					f"<figure class=\"card table\"><figcaption>Table {i} — {escape(path.name)}</figcaption><div class=\"table-wrap\">{html}</div></figure>"
				)
			body_sections.append("".join(rendered))
		else:
			body_sections.append("<p class=\"muted\">No tables found.</p>")
		body_sections.append("</section>")

		# References
		body_sections.append("<section id=\"references\" class=\"section\"><h2>References</h2>")
		if references:
			items = [f"<li>{escape(ref)}</li>" for ref in references]
			body_sections.append(f"<ol class=\"references\">{''.join(items)}</ol>")
		else:
			body_sections.append("<p class=\"muted\">No references available.</p>")
		body_sections.append("</section>")

		# Bundle
		return f"""
		<!doctype html>
		<html lang="en">
		<head>
		  <meta charset="utf-8" />
		  <meta name="viewport" content="width=device-width, initial-scale=1" />
		  <title>Hydrocortisone JSON Analysis</title>
		  <style>
			:root {{
			  --bg: #0b1020;
			  --panel: #111733;
			  --panel-2: #0f1630;
			  --text: #e8eefc;
			  --muted: #a6b1d8;
			  --accent: #7aa2ff;
			  --accent-2: #6ee7b7;
			  --border: #1d264d;
			}}
			* {{ box-sizing: border-box; }}
			body {{
			  margin: 0; padding: 0; background: var(--bg); color: var(--text);
			  font: 16px/1.6 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif;
			}}
			a {{ color: var(--accent); text-decoration: none; }}
			.banner {{ position: sticky; top: 0; z-index: 10; backdrop-filter: blur(6px);
			  background: linear-gradient(90deg, rgba(17,23,51,0.9), rgba(15,22,48,0.9)); border-bottom: 1px solid var(--border);
			  padding: 12px 20px; display: grid; gap: 4px; }}
			.banner h1 {{ margin: 0; font-size: 20px; letter-spacing: 0.3px; }}
			.banner .subtitle {{ margin: 0; color: var(--muted); font-size: 12px; }}
			.banner .nav {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px; }}
			.banner .nav a {{ padding: 4px 8px; border: 1px solid var(--border); border-radius: 8px; background: var(--panel-2); }}

			.container {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
			.section {{ margin: 24px 0; padding: 16px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; }}
			.muted {{ color: var(--muted); }}

			.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }}
			.card {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
			.card img {{ width: 100%; display: block; }}
			.card figcaption {{ font-size: 13px; color: var(--muted); padding: 8px 10px; border-top: 1px solid var(--border); }}
			.table-wrap {{ overflow-x: auto; max-width: 100%; }}

			h1, h2, h3, h4 {{ color: #fff; }}
			h2 {{ margin-top: 0; }}
			p {{ margin: 0.6em 0; }}
			.doc .header {{ color: var(--muted); font-size: 12px; margin: 0.4rem 0; }}
			.doc .title-1 {{ font-size: 28px; margin: 0.6rem 0; }}
			.doc .title-2 {{ font-size: 22px; margin: 0.6rem 0; }}
			.doc .uncategorized {{ color: var(--muted); }}
			.doc .pagebreak {{ border: 0; border-top: 1px dashed var(--border); margin: 20px 0; }}
			.pill {{ display: inline-block; padding: 2px 8px; border: 1px solid var(--border); border-radius: 999px; background: var(--panel-2); font-size: 12px; color: var(--muted); }}
			.counts {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
			.counts .pill strong {{ color: var(--text); }}
			ol.references {{ padding-left: 20px; }}
			figure.inline {{ background: var(--panel-2); border: 1px solid var(--border); padding: 8px; border-radius: 8px; margin: 12px 0; }}
			figure.inline img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
		  </style>
		</head>
		<body>
		  <div class="container">
			{''.join(body_sections)}
		  </div>
		</body>
		</html>
		"""

	def _render_summary_section(self, summary: Dict[str, Any]) -> str:
		pills = []
		for k, v in (summary.get("type_counts") or {}).items():
			pills.append(f"<span class=\"pill\"><strong>{escape(str(k))}</strong>: {escape(str(v))}</span>")
		sample_titles = summary.get("sample_titles") or []
		sample_html = "".join(f"<li>{escape(t)}</li>" for t in sample_titles)
		pages = summary.get("pages") or []
		page_info = f"Pages: {len(pages)} — {', '.join(map(str, pages[:10])) + ('…' if len(pages) > 10 else '')}" if pages else "Pages: unknown"
		return f"""
		<section id="summary" class="section">
		  <h2>Summary</h2>
		  <p class="muted">Total elements: {escape(str(summary.get('total_elements')))} — Empty text elements: {escape(str(summary.get('empty_text')))} — {escape(page_info)}</p>
		  <div class="counts">{''.join(pills)}</div>
		  {('<h3>Sample titles</h3><ol>' + sample_html + '</ol>') if sample_titles else ''}
		</section>
		"""

	def _render_document_stream(
		self,
		elements: List[Element],
		rel_images: List[Path],
		tables: List[Tuple[Path, str]],
	) -> str:
		html_parts: List[str] = ["<div class=\"doc\">"]
		img_idx, tbl_idx = 0, 0

		i = 0
		list_open = False
		title_seen = False
		while i < len(elements):
			e = elements[i]
			nxt = elements[i + 1] if i + 1 < len(elements) else None
			etype = e.type or ""
			text = (e.text or "").strip()

			# Auto-close UL when leaving list items
			if list_open and etype != "ListItem":
				html_parts.append("</ul>")
				list_open = False

			if etype == "Header":
				if text:
					html_parts.append(f"<div class=\"header\">{escape(text)}</div>")
			elif etype == "Title":
				cls = "title-1" if not title_seen else "title-2"
				html_parts.append(f"<h3 class=\"{cls}\">{escape(text) or 'Untitled'}</h3>")
				title_seen = True
			elif etype == "NarrativeText":
				if text:
					html_parts.append(f"<p>{escape(text)}</p>")
			elif etype == "UncategorizedText":
				if text:
					html_parts.append(f"<p class=\"uncategorized\">{escape(text)}</p>")
			elif etype == "ListItem":
				if not list_open:
					html_parts.append("<ul>")
					list_open = True
				if text:
					html_parts.append(f"<li>{escape(text)}</li>")
			elif etype == "FigureCaption":
				if text:
					html_parts.append(f"<p class=\"muted\"><em>{escape(text)}</em></p>")
			elif etype == "Image":
				if img_idx < len(rel_images):
					src = escape(str(rel_images[img_idx]))
					caption = None
					if nxt and nxt.type == "FigureCaption" and (nxt.text or "").strip():
						caption = nxt.text.strip()
						i += 1  # consume caption
					figcap = f"<figcaption>{escape(caption)}</figcaption>" if caption else ""
					html_parts.append(
						f"<figure class=\"inline\"><img loading=\"lazy\" src=\"{src}\" alt=\"Figure {img_idx+1}\">{figcap}</figure>"
					)
					img_idx += 1
				else:
					# No file available; drop a placeholder
					html_parts.append("<div class=\"muted\">[Image placeholder]</div>")
			elif etype == "Table":
				if tbl_idx < len(tables):
					path, tbl_html = tables[tbl_idx]
					caption = None
					if nxt and nxt.type == "FigureCaption" and (nxt.text or "").strip():
						caption = nxt.text.strip()
						i += 1
					figcap = f"<figcaption>{escape(caption)}</figcaption>" if caption else f"<figcaption>{escape(path.name)}</figcaption>"
					html_parts.append(
						f"<figure class=\"inline\"><div class=\"table-wrap\">{tbl_html}</div>{figcap}</figure>"
					)
					tbl_idx += 1
				else:
					html_parts.append("<div class=\"muted\">[Table placeholder]</div>")
			elif etype == "PageBreak":
				html_parts.append("<hr class=\"pagebreak\" />")
			else:
				# Unknown/other types
				if text:
					html_parts.append(f"<p>{escape(text)}</p>")
			i += 1

		if list_open:
			html_parts.append("</ul>")

		html_parts.append("</div>")
		return "".join(html_parts)


# ------------- Utilities


def _numeric_sort_key(p: Path) -> Tuple[int, str]:
	"""Sort by trailing integer inside filename, then lexically."""
	name = p.name
	digits = "".join(ch for ch in name if ch.isdigit())
	num = int(digits) if digits else -1
	return (num, name)


def safe_int(v: Any) -> Optional[int]:
	try:
		return int(v)
	except Exception:
		return None


def generate_html(
	input_json_path: Optional[Path | str] = None,
	output_html_path: Optional[Path | str] = None,
) -> Path:
	"""Programmatic entry point for generating the HTML report.

	Args:
		input_json_path: Optional custom input JSON path. Defaults to files/hydrocortisone-output.json.
		output_html_path: Optional custom output HTML path. Defaults to files/hydrocortisone-analysis.html.

	Returns:
		Path to the generated HTML file.
	"""
	report = HydrocortisoneHtmlReport()
	if input_json_path:
		report.input_json = Path(input_json_path)
		report.files_dir = report.input_json.parent if report.input_json.parent.name == "files" else report.files_dir
	if output_html_path:
		report.output_html = Path(output_html_path)
	return report.build()


def main() -> None:
	try:
		out = generate_html()
		logger.info("HTML report generated at: %s", out)
	except Exception as e:
		logger.exception("Failed to generate HTML: %s", e)


if __name__ == "__main__":
	main()

