"""Text Agent — handles plain text and Markdown files."""
from pathlib import Path


class TextAgent:
    """Converts plain text or Markdown files to clean Markdown output."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst"}

    def process(self, file_path: Path) -> dict:
        """
        Read a text file and return it as Markdown with metadata.

        Returns:
            {
                "markdown": str,
                "metadata": {
                    "source_file": str,
                    "file_type": str,
                    "title": str,
                }
            }
        """
        content = file_path.read_text(encoding="utf-8", errors="replace")

        if file_path.suffix == ".txt":
            markdown = self._txt_to_markdown(content, file_path.name)
        else:
            markdown = content  # .md and .rst pass through as-is

        return {
            "markdown": markdown,
            "metadata": {
                "source_file": str(file_path),
                "file_type": file_path.suffix.lstrip("."),
                "title": file_path.stem,
            },
        }

    def _txt_to_markdown(self, content: str, filename: str) -> str:
        title = filename.replace("_", " ").replace("-", " ").rsplit(".", 1)[0]
        return f"# {title}\n\n{content}"
