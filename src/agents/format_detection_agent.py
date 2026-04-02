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


# first few bytes are enough to identify most formats reliably
_MAGIC = {
    b"%PDF": FileFormat.PDF,
    b"PK\x03\x04": None,  # zip-based — could be docx/pptx/xlsx, check extension
    b"\xff\xd8\xff": FileFormat.IMAGE,  # jpeg
    b"\x89PNG": FileFormat.IMAGE,
    b"GIF8": FileFormat.IMAGE,
    b"BM": FileFormat.IMAGE,
}

_EXT = {
    ".pdf": FileFormat.PDF,
    ".docx": FileFormat.DOCX,
    ".doc": FileFormat.DOCX,
    ".pptx": FileFormat.PPTX,
    ".ppt": FileFormat.PPTX,
    ".xlsx": FileFormat.XLSX,
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
    def detect(self, file_path: Path) -> FileFormat:
        try:
            header = file_path.read_bytes()[:8]
        except OSError:
            return FileFormat.UNKNOWN

        for sig, fmt in _MAGIC.items():
            if header.startswith(sig):
                # zip-based office formats need the extension to tell apart
                return (
                    fmt
                    if fmt is not None
                    else _EXT.get(file_path.suffix.lower(), FileFormat.UNKNOWN)
                )

        return _EXT.get(file_path.suffix.lower(), FileFormat.UNKNOWN)
