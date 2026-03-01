"""Format Detection Agent — identifies true file type beyond extension."""
from enum import Enum
from pathlib import Path


class FileFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    MARKDOWN = "markdown"
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    IMAGE = "image"
    UNKNOWN = "unknown"


# Magic bytes signatures for reliable format detection
_MAGIC_SIGNATURES = {
    b"%PDF": FileFormat.PDF,
    b"PK\x03\x04": None,  # ZIP-based (Office Open XML) — disambiguate by extension
    b"\xff\xd8\xff": FileFormat.IMAGE,  # JPEG
    b"\x89PNG": FileFormat.IMAGE,  # PNG
    b"GIF8": FileFormat.IMAGE,  # GIF
    b"BM": FileFormat.IMAGE,  # BMP
}

_EXTENSION_MAP = {
    ".pdf": FileFormat.PDF,
    ".docx": FileFormat.DOCX,
    ".pptx": FileFormat.PPTX,
    ".xlsx": FileFormat.XLSX,
    ".doc": FileFormat.DOCX,
    ".ppt": FileFormat.PPTX,
    ".xls": FileFormat.XLSX,
    ".md": FileFormat.MARKDOWN,
    ".markdown": FileFormat.MARKDOWN,
    ".rst": FileFormat.TEXT,
    ".txt": FileFormat.TEXT,
    ".json": FileFormat.JSON,
    ".yaml": FileFormat.YAML,
    ".yml": FileFormat.YAML,
    ".jpg": FileFormat.IMAGE,
    ".jpeg": FileFormat.IMAGE,
    ".png": FileFormat.IMAGE,
    ".gif": FileFormat.IMAGE,
    ".bmp": FileFormat.IMAGE,
    ".tiff": FileFormat.IMAGE,
    ".webp": FileFormat.IMAGE,
}


class FormatDetectionAgent:
    """Identifies file formats using magic bytes with extension fallback."""

    def detect(self, file_path: Path) -> FileFormat:
        """
        Detect the true format of a file.

        Strategy:
        1. Read magic bytes from the file header
        2. For ZIP-based formats (Office), use extension to disambiguate
        3. Fall back to extension mapping
        4. Return UNKNOWN if nothing matches
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
        except OSError:
            return FileFormat.UNKNOWN

        # Check magic bytes
        for signature, fmt in _MAGIC_SIGNATURES.items():
            if header.startswith(signature):
                if fmt is not None:
                    return fmt
                # ZIP-based: use extension to tell docx/pptx/xlsx apart
                return _EXTENSION_MAP.get(file_path.suffix.lower(), FileFormat.UNKNOWN)

        # Fall back to extension
        return _EXTENSION_MAP.get(file_path.suffix.lower(), FileFormat.UNKNOWN)
