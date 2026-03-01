import json
from pathlib import Path

import yaml


class StructuredDataAgent:
    SUPPORTED_EXTENSIONS = {".json", ".yaml", ".yml"}

    def process(self, file_path: Path) -> dict:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        ext = file_path.suffix.lower()

        if ext == ".json":
            try:
                pretty = json.dumps(json.loads(raw), indent=2)
            except json.JSONDecodeError:
                pretty = raw
            block = f"```json\n{pretty}\n```"
            file_type = "json"
        else:
            try:
                pretty = yaml.dump(yaml.safe_load(raw), default_flow_style=False)
            except yaml.YAMLError:
                pretty = raw
            block = f"```yaml\n{pretty}\n```"
            file_type = "yaml"

        return {
            "markdown": f"# {file_path.stem}\n\n{block}\n",
            "metadata": {
                "source_file": str(file_path),
                "file_type": file_type,
                "title": file_path.stem,
            },
        }
