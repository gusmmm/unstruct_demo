from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, Text, Image, FigureCaption
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

base_dir = "./files"
pdf_file = "hydrocortisone.pdf"
pdf_file_path = f"{base_dir}/{pdf_file}"

raw_chunks = partition_pdf(
    filename=pdf_file_path,
    strategy="hi_res",
    infer_table_structure=True,
    extract_image_block_types=["Image","Figure","Table"],
    extract_image_block_to_payload=True,
    chunking_strategy=None,
    include_image_caption=True,
    include_page_number=True,
    include_table=True,
    include_table_caption=True,
    include_header_footer=True,
    include_metadata=True,
)

# Persist elements so we can reload later without re-partitioning
output_json = os.path.join(base_dir, "hydrocortisone-output.json")

try:
    # Elements from unstructured support .to_dict() for JSON-safe serialization
    serialized = [el.to_dict() for el in raw_chunks]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    logging.info("Saved %d elements to %s", len(serialized), output_json)
except Exception as e:
    logging.exception("Failed to save elements to JSON: %s", e)

# Optional: tiny helper to demonstrate how this file can be loaded later
def load_elements_from_json(path: str):
    """Load previously saved elements. Returns a list of Element objects if supported,
    otherwise returns the raw dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If Element supports from_dict (newer unstructured), reconstruct Element objects
    if hasattr(Element, "from_dict"):
        return [Element.from_dict(d) for d in data]
    return data

logging.info("Number of elements: %d", len(raw_chunks))
for i, element in enumerate(raw_chunks):
    logging.debug("Element %d: %s", i, element)
