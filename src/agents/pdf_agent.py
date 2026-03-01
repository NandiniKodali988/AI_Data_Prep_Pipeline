"""PDF Agent — extracts text, tables, and images from PDF files."""
import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from src.agents.image_processing_agent import ImageProcessingAgent

logger = logging.getLogger(__name__)


class PDFAgent:
    """
    Processes PDF files into Markdown.

    Pipeline per PDF:
    1. Extract text per page using PyMuPDF (preserves reading order)
    2. Extract tables per page using pdfplumber → Markdown tables
    3. Extract embedded images → describe with Claude Vision
    4. Assemble into a single Markdown document with page markers
    """

    def __init__(self, image_agent: ImageProcessingAgent, images_dir: Path):
        """
        Args:
            image_agent: ImageProcessingAgent instance for vision descriptions.
            images_dir: Directory where extracted images will be saved.
        """
        self.image_agent = image_agent
        self.images_dir = images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def process(self, file_path: Path) -> dict:
        """
        Process a PDF and return Markdown + metadata.

        Returns:
            {
                "markdown": str,
                "metadata": {
                    "source_file": str,
                    "file_type": "pdf",
                    "title": str,
                    "author": str,
                    "page_count": int,
                    "creation_date": str,
                }
            }
        """
        metadata = self._extract_metadata(file_path)
        pages_md = []

        with fitz.open(str(file_path)) as doc, pdfplumber.open(str(file_path)) as plumber_doc:
            for page_num, (fitz_page, plumber_page) in enumerate(
                zip(doc, plumber_doc.pages), start=1
            ):
                page_md = self._process_page(
                    fitz_page=fitz_page,
                    plumber_page=plumber_page,
                    page_num=page_num,
                    doc_title=metadata["title"],
                    doc=doc,
                )
                pages_md.append(page_md)

        title = metadata["title"]
        author = metadata.get("author", "")
        header = f"# {title}\n"
        if author:
            header += f"\n**Author:** {author}\n"
        header += "\n---\n\n"

        full_markdown = header + "\n\n".join(pages_md)

        return {"markdown": full_markdown, "metadata": metadata}

    def _process_page(self, fitz_page, plumber_page, page_num: int, doc_title: str, doc) -> str:
        parts = [f"## Page {page_num}\n"]

        # 1. Text extraction
        text = fitz_page.get_text("text").strip()
        if text:
            parts.append(text)

        # 2. Table extraction (replaces raw text for tabular content)
        tables = plumber_page.extract_tables()
        if tables:
            for table in tables:
                parts.append(self._table_to_markdown(table))

        # 3. Image extraction
        image_list = fitz_page.get_images(full=True)
        for img_index, img_ref in enumerate(image_list):
            xref = img_ref[0]
            try:
                image_md = self._extract_and_describe_image(
                    doc=doc,
                    xref=xref,
                    page_num=page_num,
                    img_index=img_index,
                    doc_title=doc_title,
                )
                parts.append(image_md)
            except Exception as e:
                logger.warning("Failed to extract image xref=%d on page %d: %s", xref, page_num, e)

        return "\n\n".join(parts)

    def _extract_and_describe_image(
        self, doc, xref: int, page_num: int, img_index: int, doc_title: str
    ) -> str:
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        image_filename = f"p{page_num}_img{img_index}.{image_ext}"
        image_path = self.images_dir / image_filename
        image_path.write_bytes(image_bytes)

        context = f"a PDF document titled '{doc_title}' (page {page_num})"
        return self.image_agent.describe(image_path, document_context=context)

    def _extract_metadata(self, file_path: Path) -> dict:
        with fitz.open(str(file_path)) as doc:
            meta = doc.metadata or {}
            return {
                "source_file": str(file_path),
                "file_type": "pdf",
                "title": meta.get("title") or file_path.stem,
                "author": meta.get("author", ""),
                "creation_date": meta.get("creationDate", ""),
                "page_count": doc.page_count,
            }

    def _table_to_markdown(self, table: list[list]) -> str:
        if not table or not table[0]:
            return ""

        header = table[0]
        rows = table[1:]

        def cell(v):
            return str(v).replace("\n", " ").strip() if v is not None else ""

        md = "| " + " | ".join(cell(h) for h in header) + " |\n"
        md += "| " + " | ".join("---" for _ in header) + " |\n"
        for row in rows:
            # Pad short rows
            padded = row + [None] * (len(header) - len(row))
            md += "| " + " | ".join(cell(c) for c in padded) + " |\n"

        return md
