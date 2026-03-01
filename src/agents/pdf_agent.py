import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from src.agents.image_processing_agent import ImageProcessingAgent

logger = logging.getLogger(__name__)


class PDFAgent:
    def __init__(self, image_agent: ImageProcessingAgent, images_dir: Path):
        self.image_agent = image_agent
        self.images_dir = images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def process(self, file_path: Path) -> dict:
        metadata = self._get_metadata(file_path)
        pages_md = []

        with fitz.open(str(file_path)) as doc, pdfplumber.open(str(file_path)) as plumber:
            for page_num, (fitz_page, plumber_page) in enumerate(zip(doc, plumber.pages), start=1):
                pages_md.append(self._process_page(fitz_page, plumber_page, page_num, metadata["title"], doc))

        header = f"# {metadata['title']}\n"
        if metadata.get("author"):
            header += f"\n**Author:** {metadata['author']}\n"
        header += "\n---\n\n"

        return {"markdown": header + "\n\n".join(pages_md), "metadata": metadata}

    def _process_page(self, fitz_page, plumber_page, page_num, doc_title, doc) -> str:
        parts = [f"## Page {page_num}\n"]

        text = fitz_page.get_text("text").strip()
        if text:
            parts.append(text)

        for table in plumber_page.extract_tables() or []:
            parts.append(self._table_to_markdown(table))

        for i, img_ref in enumerate(fitz_page.get_images(full=True)):
            xref = img_ref[0]
            try:
                parts.append(self._extract_image(doc, xref, page_num, i, doc_title))
            except Exception as e:
                logger.warning("page %d image %d failed: %s", page_num, i, e)

        return "\n\n".join(parts)

    def _extract_image(self, doc, xref, page_num, img_index, doc_title) -> str:
        raw = doc.extract_image(xref)
        img_path = self.images_dir / f"p{page_num}_img{img_index}.{raw['ext']}"
        img_path.write_bytes(raw["image"])
        return self.image_agent.describe(img_path, f"a PDF titled '{doc_title}' (page {page_num})")

    def _get_metadata(self, file_path: Path) -> dict:
        with fitz.open(str(file_path)) as doc:
            m = doc.metadata or {}
            return {
                "source_file": str(file_path),
                "file_type": "pdf",
                "title": m.get("title") or file_path.stem,
                "author": m.get("author", ""),
                "creation_date": m.get("creationDate", ""),
                "page_count": doc.page_count,
            }

    def _table_to_markdown(self, table: list) -> str:
        if not table or not table[0]:
            return ""

        def cell(v):
            return str(v).replace("\n", " ").strip() if v is not None else ""

        header = table[0]
        md = "| " + " | ".join(cell(h) for h in header) + " |\n"
        md += "| " + " | ".join("---" for _ in header) + " |\n"
        for row in table[1:]:
            padded = row + [None] * (len(header) - len(row))
            md += "| " + " | ".join(cell(c) for c in padded) + " |\n"
        return md
