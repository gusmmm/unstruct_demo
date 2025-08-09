"""
Summarize Image elements for RAG: read images_llm_dump.json, send image + caption + text
to Google GenAI, and save a per-image summary JSON.

Usage:
  uv run iterations/images_summary.py \
    --input ./files/extracted/images/images_llm_dump.json \
    --out ./files/extracted/images/images_llm_summaries.json \
    --model gemini-2.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("images_summary")


def load_dump(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of image entries")
    return data


def summarize_one(
    client: genai.Client,
    model: str,
    element_id: str,
    caption: Optional[str],
    image_text: Optional[str],
    image_base64: Optional[str],
    image_mime_type: Optional[str],
    max_tokens: int = 250,
) -> str:
    """Call Google GenAI (google-genai) to summarize one image with provided context."""
    parts: List[Dict[str, Any]] = []

    instructions = (
        "You are assisting with a clinical research paper analysis. "
        "Summarize the key information depicted in the image in 3-6 concise sentences. "
        "Include concrete details (axes/labels/units if visible, key patterns or counts), "
        "and how it relates to the study (trial flow, results, outcomes) when possible. "
        "Avoid speculation. Max 250 tokens."
    )
    parts.append({"text": instructions})

    ctx_lines = []
    if caption:
        ctx_lines.append(f"Caption: {caption}")
    if image_text:
        ctx_lines.append(f"On-image text: {image_text}")
    if ctx_lines:
        parts.append({"text": "\n".join(ctx_lines)})

    if image_base64 and image_mime_type:
        parts.append(
            {
                "inline_data": {
                    "mime_type": image_mime_type,
                    "data": image_base64,
                }
            }
        )
    else:
        parts.append({"text": "No binary image available; summarize based on caption and text only."})

    # google-genai Client usage: models.generate_content with contents and generation_config
    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": parts}],
        config={
            "max_output_tokens": max_tokens,
            "temperature": 0.2,
        },
    )

    # Extract text from response
    # The google-genai response typically has .text or .output_text
    text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    # Some responses may have .candidates list with .content.parts[].text
    cand = getattr(resp, "candidates", None)
    if cand:
        for c in cand:
            content = getattr(c, "content", None) or {}
            parts_out = getattr(content, "parts", None) or []
            for p in parts_out:
                if isinstance(p, dict):
                    t = p.get("text")
                    if t:
                        return str(t).strip()
                else:
                    t = getattr(p, "text", None)
                    if t:
                        return str(t).strip()
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize image elements via Google GenAI")
    p.add_argument("--input", type=Path, default=Path("./files/extracted/images/images_llm_dump.json"))
    p.add_argument("--out", type=Path, default=Path("./files/extracted/images/images_llm_summaries.json"))
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--max-tokens", type=int, default=250)
    return p.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    items = load_dump(args.input)
    logger.info("Loaded %d image entries", len(items))

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        element_id = item.get("element_id")
        caption = item.get("caption")
        image_text = item.get("image_text")
        image_b64 = item.get("image_base64")
        mime = item.get("image_mime_type")

        logger.info("Summarizing %d/%d: %s", idx, len(items), element_id)
        try:
            summary = summarize_one(
                client,
                args.model,
                element_id=element_id,
                caption=caption,
                image_text=image_text,
                image_base64=image_b64,
                image_mime_type=mime,
                max_tokens=args.max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("GenAI call failed for %s: %s", element_id, e)
            summary = ""

        results.append(
            {
                "element_id": element_id,
                "summary": summary,
                "model": args.model,
                "has_image": bool(image_b64),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d summaries to %s", len(results), args.out)


if __name__ == "__main__":
    main()

