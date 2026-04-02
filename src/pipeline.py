import logging
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.agents.chunking_agent import ChunkingAgent
from src.agents.format_detection_agent import FileFormat, FormatDetectionAgent
from src.agents.image_processing_agent import ImageProcessingAgent
from src.agents.indexing_agent import IndexingAgent
from src.agents.metadata_agent import MetadataAgent
from src.agents.office_agent import DocxAgent, PptxAgent, XlsxAgent
from src.agents.pdf_agent import PDFAgent
from src.agents.structured_data_agent import StructuredDataAgent
from src.agents.text_agent import TextAgent

load_dotenv()
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, output_dir: Path, chroma_db_path: str = "./chroma_db"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        client = anthropic.Anthropic()
        images_dir = output_dir / "images"

        self.format_agent = FormatDetectionAgent()
        self.image_agent = ImageProcessingAgent(client=client)
        self.pdf_agent = PDFAgent(image_agent=self.image_agent, images_dir=images_dir)
        self.docx_agent = DocxAgent(image_agent=self.image_agent, images_dir=images_dir)
        self.pptx_agent = PptxAgent(image_agent=self.image_agent, images_dir=images_dir)
        self.xlsx_agent = XlsxAgent()
        self.text_agent = TextAgent()
        self.structured_agent = StructuredDataAgent()
        self.metadata_agent = MetadataAgent()
        self.chunking_agent = ChunkingAgent()
        self.indexing_agent = IndexingAgent(chroma_db_path=chroma_db_path)

    def run(self, input_dir: Path) -> dict[str, int | list[str]]:
        files = [f for f in input_dir.rglob("*") if f.is_file()]
        processed, skipped, failed, total_chunks = 0, 0, 0, 0
        processed_files = []

        for file_path in files:
            fmt = self.format_agent.detect(file_path)
            if fmt == FileFormat.UNKNOWN:
                logger.info("skipping %s (unsupported)", file_path.name)
                skipped += 1
                continue

            logger.info("[%s] %s", fmt.value, file_path.name)
            try:
                result = self._process_file(file_path, fmt)
                if result is None:
                    skipped += 1
                    continue

                out_path = self.output_dir / (file_path.stem + ".md")
                out_path.write_text(result["markdown"], encoding="utf-8")

                chunks = self.chunking_agent.chunk(result["markdown"], result["metadata"])
                n = self.indexing_agent.index(chunks)
                total_chunks += n
                processed += 1
                processed_files.append(str(file_path))
                logger.info("  %d chunks from %s", n, file_path.name)

            except Exception as e:
                logger.error("%s failed: %s", file_path.name, e, exc_info=True)
                failed += 1

        return {
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "total_chunks": total_chunks,
            "files": processed_files,
        }

    def process_single(self, file_path: Path) -> dict[str, str | int] | None:
        """Process and index one file. Returns {chunks, format} or None if skipped/failed."""
        fmt = self.format_agent.detect(file_path)
        if fmt == FileFormat.UNKNOWN:
            logger.info("skipping %s (unsupported format)", file_path.name)
            return None
        try:
            result = self._process_file(file_path, fmt)
            if result is None:
                return None
            out_path = self.output_dir / (file_path.stem + ".md")
            out_path.write_text(result["markdown"], encoding="utf-8")
            chunks = self.chunking_agent.chunk(result["markdown"], result["metadata"])
            n = self.indexing_agent.index(chunks)
            logger.info("indexed %d chunks from %s", n, file_path.name)
            return {"chunks": n, "format": fmt.value}
        except Exception as e:
            logger.error("%s failed: %s", file_path.name, e, exc_info=True)
            raise

    def _process_file(self, file_path: Path, fmt: FileFormat) -> dict | None:
        if fmt == FileFormat.PDF:
            result = self.pdf_agent.process(file_path)
        elif fmt == FileFormat.DOCX:
            result = self.docx_agent.process(file_path)
        elif fmt == FileFormat.PPTX:
            result = self.pptx_agent.process(file_path)
        elif fmt == FileFormat.XLSX:
            result = self.xlsx_agent.process(file_path)
        elif fmt in (FileFormat.TEXT, FileFormat.MARKDOWN):
            result = self.text_agent.process(file_path)
        elif fmt in (FileFormat.JSON, FileFormat.YAML):
            result = self.structured_agent.process(file_path)
        elif fmt == FileFormat.IMAGE:
            desc = self.image_agent.describe(file_path, f"a standalone image '{file_path.name}'")
            result = {
                "markdown": f"# {file_path.stem}\n\n{desc}",
                "metadata": {
                    "source_file": str(file_path),
                    "file_type": "image",
                    "title": file_path.stem,
                },
            }
        else:
            return None

        result["metadata"] = self.metadata_agent.enrich(result["metadata"], file_path)
        return result
