"""Pipeline orchestrator — routes files through the appropriate agents."""
import logging
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.agents.chunking_agent import ChunkingAgent
from src.agents.format_detection_agent import FileFormat, FormatDetectionAgent
from src.agents.image_processing_agent import ImageProcessingAgent
from src.agents.indexing_agent import IndexingAgent
from src.agents.metadata_agent import MetadataAgent
from src.agents.pdf_agent import PDFAgent
from src.agents.structured_data_agent import StructuredDataAgent
from src.agents.text_agent import TextAgent

load_dotenv()
logger = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end document processing pipeline.

    Usage:
        pipeline = Pipeline(output_dir=Path("./output"))
        results = pipeline.run(input_dir=Path("./data"))
    """

    def __init__(self, output_dir: Path, chroma_db_path: str = "./chroma_db"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"

        anthropic_client = anthropic.Anthropic()

        self.format_agent = FormatDetectionAgent()
        self.image_agent = ImageProcessingAgent(client=anthropic_client)
        self.pdf_agent = PDFAgent(image_agent=self.image_agent, images_dir=images_dir)
        self.text_agent = TextAgent()
        self.structured_agent = StructuredDataAgent()
        self.metadata_agent = MetadataAgent()
        self.chunking_agent = ChunkingAgent()
        self.indexing_agent = IndexingAgent(chroma_db_path=chroma_db_path)

    def run(self, input_dir: Path) -> dict:
        """
        Process all supported files in input_dir.

        Returns:
            Summary dict: {
                "processed": int,
                "skipped": int,
                "failed": int,
                "total_chunks": int,
                "files": list[str]
            }
        """
        files = [f for f in input_dir.rglob("*") if f.is_file()]
        processed, skipped, failed, total_chunks = 0, 0, 0, 0
        processed_files = []

        for file_path in files:
            fmt = self.format_agent.detect(file_path)
            if fmt == FileFormat.UNKNOWN:
                logger.info("Skipping unsupported file: %s", file_path.name)
                skipped += 1
                continue

            logger.info("Processing [%s] %s", fmt.value, file_path.name)
            try:
                result = self._process_file(file_path, fmt)
                if result is None:
                    skipped += 1
                    continue

                markdown, metadata = result["markdown"], result["metadata"]

                # Write Markdown output
                out_path = self.output_dir / (file_path.stem + ".md")
                out_path.write_text(markdown, encoding="utf-8")

                # Chunk and index
                chunks = self.chunking_agent.chunk(markdown, metadata)
                n = self.indexing_agent.index(chunks)
                total_chunks += n

                processed += 1
                processed_files.append(str(file_path))
                logger.info("  → %d chunks indexed from %s", n, file_path.name)

            except Exception as e:
                logger.error("Failed to process %s: %s", file_path.name, e, exc_info=True)
                failed += 1

        return {
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "total_chunks": total_chunks,
            "files": processed_files,
        }

    def _process_file(self, file_path: Path, fmt: FileFormat) -> dict | None:
        if fmt == FileFormat.PDF:
            result = self.pdf_agent.process(file_path)
        elif fmt in (FileFormat.TEXT, FileFormat.MARKDOWN):
            result = self.text_agent.process(file_path)
        elif fmt in (FileFormat.JSON, FileFormat.YAML):
            result = self.structured_agent.process(file_path)
        elif fmt == FileFormat.IMAGE:
            # Standalone image — describe it directly
            context = f"a standalone image file named '{file_path.name}'"
            description_md = self.image_agent.describe(file_path, document_context=context)
            result = {
                "markdown": f"# {file_path.stem}\n\n{description_md}",
                "metadata": {"source_file": str(file_path), "file_type": "image", "title": file_path.stem},
            }
        else:
            logger.info("No agent available for format %s (%s)", fmt.value, file_path.name)
            return None

        result["metadata"] = self.metadata_agent.enrich(result["metadata"], file_path)
        return result
