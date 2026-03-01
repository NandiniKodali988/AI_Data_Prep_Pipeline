from pathlib import Path


class TextAgent:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst"}

    def process(self, file_path: Path) -> dict:
        content = file_path.read_text(encoding="utf-8", errors="replace")

        if file_path.suffix == ".txt":
            title = file_path.stem.replace("_", " ").replace("-", " ")
            markdown = f"# {title}\n\n{content}"
        else:
            markdown = content

        return {
            "markdown": markdown,
            "metadata": {
                "source_file": str(file_path),
                "file_type": file_path.suffix.lstrip("."),
                "title": file_path.stem,
            },
        }
