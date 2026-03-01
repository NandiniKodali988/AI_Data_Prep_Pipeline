"""Tests for FormatDetectionAgent."""
import struct
import tempfile
from pathlib import Path

import pytest

from src.agents.format_detection_agent import FileFormat, FormatDetectionAgent


@pytest.fixture
def agent():
    return FormatDetectionAgent()


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


class TestFormatDetectionAgent:
    def test_detects_pdf_by_magic_bytes(self, agent, tmp):
        f = tmp / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        assert agent.detect(f) == FileFormat.PDF

    def test_detects_json_by_extension(self, agent, tmp):
        f = tmp / "data.json"
        f.write_text('{"key": "value"}')
        assert agent.detect(f) == FileFormat.JSON

    def test_detects_yaml_by_extension(self, agent, tmp):
        f = tmp / "config.yaml"
        f.write_text("key: value")
        assert agent.detect(f) == FileFormat.YAML

    def test_detects_markdown(self, agent, tmp):
        f = tmp / "readme.md"
        f.write_text("# Title")
        assert agent.detect(f) == FileFormat.MARKDOWN

    def test_detects_plain_text(self, agent, tmp):
        f = tmp / "notes.txt"
        f.write_text("some text")
        assert agent.detect(f) == FileFormat.TEXT

    def test_detects_png_by_magic(self, agent, tmp):
        f = tmp / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert agent.detect(f) == FileFormat.IMAGE

    def test_unknown_extension_returns_unknown(self, agent, tmp):
        f = tmp / "weird.xyz"
        f.write_bytes(b"\x00\x01\x02\x03")
        assert agent.detect(f) == FileFormat.UNKNOWN

    def test_missing_file_returns_unknown(self, agent, tmp):
        f = tmp / "nonexistent.pdf"
        assert agent.detect(f) == FileFormat.UNKNOWN
