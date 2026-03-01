"""Structured Data Agent — converts JSON and YAML to Markdown code blocks."""
import json
from pathlib import Path

import yaml


class StructuredDataAgent:
    """Converts JSON/YAML files into readable Markdown code blocks."""

    SUPPORTED_EXTENSIONS = {".json", ".yaml", ".yml"}

    def process(self, file_path: Path) -> dict:
        """
        Read a JSON/YAML file and return it as a Markdown code block.

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
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        ext = file_path.suffix.lower()

        if ext == ".json":
            markdown = self._json_to_markdown(raw, file_path.stem)
            file_type = "json"
        else:
            markdown = self._yaml_to_markdown(raw, file_path.stem)
            file_type = "yaml"

        return {
            "markdown": markdown,
            "metadata": {
                "source_file": str(file_path),
                "file_type": file_type,
                "title": file_path.stem,
            },
        }

    def _json_to_markdown(self, raw: str, title: str) -> str:
        try:
            parsed = json.loads(raw)
            pretty = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pretty = raw  # fall back to raw if malformed
        return f"# {title}\n\n```json\n{pretty}\n```\n"

    def _yaml_to_markdown(self, raw: str, title: str) -> str:
        try:
            parsed = yaml.safe_load(raw)
            pretty = yaml.dump(parsed, default_flow_style=False)
        except yaml.YAMLError:
            pretty = raw
        return f"# {title}\n\n```yaml\n{pretty}\n```\n"
