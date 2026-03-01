from pathlib import Path


class MetadataAgent:
    def enrich(self, metadata: dict, file_path: Path) -> dict:
        m = dict(metadata)
        m.setdefault("source_file", str(file_path))
        m.setdefault("file_type", file_path.suffix.lstrip("."))
        m.setdefault("title", file_path.stem)
        m.setdefault("author", "")
        m.setdefault("creation_date", "")
        m.setdefault("page_count", 0)
        m["source_file"] = str(Path(m["source_file"]).resolve())
        m["filename"] = Path(m["source_file"]).name
        return m
