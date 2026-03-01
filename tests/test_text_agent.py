"""Tests for TextAgent and StructuredDataAgent."""
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.agents.structured_data_agent import StructuredDataAgent
from src.agents.text_agent import TextAgent


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


class TestTextAgent:
    def setup_method(self):
        self.agent = TextAgent()

    def test_plain_text_becomes_markdown(self, tmp):
        f = tmp / "hello.txt"
        f.write_text("Hello world")
        result = self.agent.process(f)
        assert "# hello" in result["markdown"]
        assert "Hello world" in result["markdown"]

    def test_markdown_passthrough(self, tmp):
        content = "# My Doc\n\nSome text."
        f = tmp / "doc.md"
        f.write_text(content)
        result = self.agent.process(f)
        assert result["markdown"] == content

    def test_metadata_has_required_fields(self, tmp):
        f = tmp / "notes.txt"
        f.write_text("test")
        result = self.agent.process(f)
        meta = result["metadata"]
        assert meta["file_type"] == "txt"
        assert meta["title"] == "notes"
        assert "source_file" in meta


class TestStructuredDataAgent:
    def setup_method(self):
        self.agent = StructuredDataAgent()

    def test_json_wrapped_in_code_block(self, tmp):
        data = {"key": "value", "count": 42}
        f = tmp / "config.json"
        f.write_text(json.dumps(data))
        result = self.agent.process(f)
        assert "```json" in result["markdown"]
        assert '"key"' in result["markdown"]

    def test_yaml_wrapped_in_code_block(self, tmp):
        data = {"name": "test", "enabled": True}
        f = tmp / "settings.yaml"
        f.write_text(yaml.dump(data))
        result = self.agent.process(f)
        assert "```yaml" in result["markdown"]
        assert "name" in result["markdown"]

    def test_malformed_json_does_not_crash(self, tmp):
        f = tmp / "bad.json"
        f.write_text("{not valid json")
        result = self.agent.process(f)
        assert "```json" in result["markdown"]
