## Goal

Build an end-to-end RAG pipeline over scientific PDFs using the artifacts produced by this repo (text, images, tables, references), store them in ChromaDB with Gemini embeddings, and expose a Google GenAI agent team with session memory, long-term memory backed by the vector index, and PKM tools.

## Prerequisites

- Python 3.12 with uv for dependency and script execution.
- Environment variables in `.env`:
	- `GEMINI_API_KEY=...`
	- Optional: `CHROMA_DB_DIR=./.chroma` (default local folder for Chroma persistence)
- Dependencies (add if missing): `google-genai`, `chromadb`, `python-dotenv`, `unstructured` and parsers used in this repo.

## Current artifacts (from this repo)

- Persisted elements (Unstructured): `files/hydrocortisone-output.json`
- Images dump: `files/extracted/images/images_llm_dump.json`
- Image summaries: `files/extracted/images/images_llm_summaries.json`
- Tables dump: `files/extracted/tables/tables_llm_dump.json`
- Table summaries: `files/extracted/tables/tables_llm_summaries.json`
- References: `files/hydrocortisone-references.json` and `files/hydrocortisone-references.txt`

## Index design (ChromaDB collections)

Create separate collections to keep semantics clear while allowing cross-source retrieval:

1. `papers_text` — narrative text chunks
	 - id: element_id
	 - text: chunk text
	 - metadata: { doc_id, page_number, title, section, source_path, element_type, parent_id }

2. `tables` — table facts and structure
	 - id: element_id
	 - text: preferred embedding text in priority order:
		 - table summary (from `tables_llm_summaries.json`) if present
		 - else `table_text`
		 - optionally append a compacted HTML→Markdown snippet
	 - metadata: { caption, page_number, table_html_file, doc_id, source_path }

3. `figures` — image content
	 - id: element_id
	 - text: image summary (from `images_llm_summaries.json`) plus caption and any OCR’d on-image text
	 - metadata: { caption, page_number, image_path (if any), doc_id, source_path }

4. `references` — bibliography/links
	 - id: stable hash of citation string or DOI/URL
	 - text: citation string + abstract/notes if available
	 - metadata: { doi, url, authors, year, venue }

5. `notes` (PKM) — user notes and extracted insights
	 - id: generated UUID
	 - text: note text
	 - metadata: { tags, created_at, source_ids (tables/figures/text), session_id }

## Step-by-step plan

1) Prepare extraction artifacts (already implemented in this repo)
- Persist elements: run `iterations/V2_pieces.py` to produce `hydrocortisone-output.json`.
- Extract images dump + summarize: `iterations/analyse_images.py` then `iterations/images_summary.py`.
- Extract tables dump + summarize: `iterations/analyse_tables.py` then `iterations/tables_summary.py`.
- Extract references (available in `files/hydrocortisone-references.*`).

2) Normalize data for indexing
- Create a small ingestion script that:
	- Loads `hydrocortisone-output.json` and selects narrative/textual `Element` types (e.g., NarrativeText, Title, ListItem) and prepares chunk records for `papers_text`.
	- Loads `tables_llm_dump.json` and joins with `tables_llm_summaries.json` by `element_id` to produce table records.
	- Loads `images_llm_dump.json` and joins with `images_llm_summaries.json` by `element_id` to produce figure records.
	- Loads references JSON/TXT, parsing into citation records.
	- Writes temporary normalized JSON lines (one file per collection) under `./files/normalized/` for auditable ingestion.

3) Configure Gemini embeddings
- Model: `text-embedding-004` (recommended general-purpose embedding model).
- Batch size: 32–96 texts per call (adjust to rate limits); retry with backoff on 429/5xx.
- Define a reusable embedding function for Chroma:
	- Inputs: List[str]
	- Output: List[List[float]] (one vector per text)
	- Implementation: call Gemini embeddings API with the batch of texts; on errors, retry; for empty/None texts, skip or substitute a minimal placeholder.

4) Initialize ChromaDB (persistent)
- Directory: `CHROMA_DB_DIR` (default `./.chroma`).
- For each collection name above, create if not exists with the custom embedding function bound.

5) Upsert data into Chroma
- For each normalized file:
	- Read ids, texts, metadatas.
	- Upsert in batches to the corresponding collection.
	- Log counts and any skipped rows (e.g., empty text after cleaning).

6) Retrieval pipeline (RAG)
- Query flow:
	1. Receive user query.
	2. Expand/clarify (optional): ask a lightweight Gemini call to generate 1–3 reformulations.
	3. Embed original query (and optionally expansions) with Gemini embeddings.
	4. Search top-k across collections (`papers_text`, `tables`, `figures`, `references`) with Chroma `query`.
	5. Merge and de-duplicate results by document and section; include diverse types (table, figure, text) for balanced context.
	6. Build the final context window:
		 - Include citations: captions, page numbers, and source paths.
		 - Prefer table/figure summaries as compact evidence.
	7. Ask a Gemini generation model (e.g., `gemini-2.0-flash`/`gemini-2.5-flash-lite`) to synthesize an answer citing sources.

7) Google GenAI agent team (high-level design)
- Orchestrator Agent
	- Routes intents: ingest, search, answer, save note, cite sources.
- Ingestion Agent
	- Runs the extractors (existing scripts) and normalizer; monitors new PDFs dropped into `files/`.
- Indexer Agent
	- Computes embeddings, upserts to Chroma, validates counts.
- Retrieval & Synthesis Agent
	- Executes the retrieval pipeline and drafts grounded answers with source attributions.
- Memory Agent
	- Session memory: stores conversation history and rolling summaries per session_id (in a light store or Chroma `notes`).
	- Long-term memory: writes important facts/insights as `notes` records and embeds them.
- PKM Tools
	- Add Note: persist a note and index it into `notes`.
	- Save Source: attach current citations to a note with tags.
	- Export: write selected notes to Markdown under `./notes/` (git-friendly) and keep indexed.

8) Prompt and grounding patterns
- System instructions: always cite page numbers and element_ids when possible.
- Provide strict answer format: summary → bullet evidence lines with [type: id, page, caption/title].
- Include a “No answer” fallback when retrieval confidence is low (few/no matches).

9) Validation & monitoring
- Index checks: number of embedded items per collection vs. normalized rows.
- Spot queries: known-answer prompts should retrieve the relevant table/figure/text.
- Logging: use Python logging; capture rate-limit retries and any embedding failures.

10) Maintenance
- Periodic re-embedding on model upgrades.
- Vacuum/compact Chroma as collections grow.
- Schema evolution: allow adding new metadata keys; keep ingestion code tolerant to missing fields.

## Minimal implementation checklist

1. Ensure env and deps
	 - `.env` has `GEMINI_API_KEY`
	 - Install deps with uv (if missing): `uv add google-genai chromadb python-dotenv`

2. Produce artifacts (if not already present)
	 - Run existing scripts in `iterations/` to refresh dumps and summaries.

3. Write `ingest_to_chroma.py` (outline)
	 - Load normalized inputs; define `GeminiEmbeddingFunction`; create/open Chroma client with `persist_directory=CHROMA_DB_DIR`.
	 - Create collections and upsert in batches with vectors via the embedding function.

4. Write `rag_query.py` (outline)
	 - Given a query, embed with Gemini; query multiple collections; assemble context; call Gemini for grounded answer; print answer with citations.

5. Wire the Agent team
	 - Create a lightweight orchestrator that exposes commands: ingest, index, search, answer, save-note.
	 - Persist session memory and offer a tool to promote messages → long-term notes.

## Data contracts (per collection)

All collections use: ids: List[str], embeddings: via Gemini, metadatas: Dict[str, Any]

- papers_text
	- id: element_id
	- text: textual chunk (<= 2–4k chars recommended)
	- metadata: { doc_id, page_number, title, section, source_path, element_type, parent_id }

- tables
	- id: element_id
	- text: summary or table_text (+ optional md from HTML)
	- metadata: { caption, page_number, table_html_file, doc_id, source_path }

- figures
	- id: element_id
	- text: image summary + caption (+ OCR text if available)
	- metadata: { caption, page_number, image_path, doc_id, source_path }

- references
	- id: hash(doi|url|citation)
	- text: citation string (+ abstract if present)
	- metadata: { doi, url, authors, year, venue }

- notes
	- id: uuid
	- text: note text
	- metadata: { tags, created_at, source_ids, session_id }

## Try it (commands)

```bash
# 1) (Optional) install missing deps
uv add google-genai chromadb python-dotenv

# 2) Summarize tables (already added)
uv run iterations/tables_summary.py \
	--input files/extracted/tables/tables_llm_dump.json \
	--out files/extracted/tables/tables_llm_summaries.json \
	--model gemini-2.5-flash-lite \
	--max-tokens 250 \
	--html-max-chars 8000

# 3) Summarize images (already added)
uv run iterations/images_summary.py \
	--input files/extracted/images/images_llm_dump.json \
	--out files/extracted/images/images_llm_summaries.json \
	--model gemini-2.5-flash-lite \
	--max-tokens 250

# 4) (Next) Implement and run: ingest_to_chroma.py
#    - creates collections and upserts vectors using Gemini embeddings

# 5) (Next) Implement and run: rag_query.py
#    - queries Chroma across collections and answers with grounded citations
```

## Notes

- Prefer concise, summary-like text for embeddings (table/figure summaries are ideal).
- Use chunk-by-title for text to preserve section context.
- Keep provenance: always store page numbers, element_ids, and filenames in metadata for citations.

