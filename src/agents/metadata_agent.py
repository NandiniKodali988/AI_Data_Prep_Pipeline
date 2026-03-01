"""Metadata Agent — enriches document metadata for provenance and search."""
from pathlib import Path


class MetadataAgent:
    """
    Normalizes and enriches metadata from any document processor's output.

    Each processed document carries a metadata dict. This agent ensures a
    consistent schema across all formats so the indexing and chunking agents
    can rely on a stable structure.
    """

    REQUIRED_FIELDS = {"source_file", "file_type", "title"}

    def enrich(self, metadata: dict, file_path: Path) -> dict:
        """
        Fill in missing fields and normalize values.

        Args:
            metadata: Raw metadata dict from a document agent.
            file_path: Original file path (used to fill gaps).

        Returns:
            Normalized metadata dict with guaranteed fields.
        """
        enriched = dict(metadata)

        # Ensure required fields have values
        enriched.setdefault("source_file", str(file_path))
        enriched.setdefault("file_type", file_path.suffix.lstrip("."))
        enriched.setdefault("title", file_path.stem)
        enriched.setdefault("author", "")
        enriched.setdefault("creation_date", "")
        enriched.setdefault("page_count", 0)

        # Normalize source_file to absolute string
        enriched["source_file"] = str(Path(enriched["source_file"]).resolve())

        # Derive filename for easy display
        enriched["filename"] = Path(enriched["source_file"]).name

        return enriched
