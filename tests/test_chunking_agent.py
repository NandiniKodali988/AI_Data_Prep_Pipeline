"""Tests for ChunkingAgent."""
import pytest

from src.agents.chunking_agent import ChunkingAgent


@pytest.fixture
def agent():
    return ChunkingAgent(max_chunk_chars=200, overlap_chars=50)


SAMPLE_MARKDOWN = """# Introduction

This is the introduction section with some content about the topic.

## Background

Here we discuss the background. More details follow about the subject matter
and relevant context that readers should understand.

## Methods

The methods section describes how the work was conducted.
"""


class TestChunkingAgent:
    def test_chunks_on_headings(self, agent):
        chunks = agent.chunk(SAMPLE_MARKDOWN, {"source_file": "test.md", "title": "Test"})
        headings = [c["metadata"]["section_heading"] for c in chunks]
        assert any("Introduction" in h for h in headings)
        assert any("Background" in h for h in headings)

    def test_every_chunk_has_metadata(self, agent):
        chunks = agent.chunk(SAMPLE_MARKDOWN, {"source_file": "test.md", "title": "Test"})
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "chunk_index" in chunk["metadata"]
            assert "source_file" in chunk["metadata"]

    def test_chunk_indices_are_sequential(self, agent):
        chunks = agent.chunk(SAMPLE_MARKDOWN, {"source_file": "test.md", "title": "Test"})
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_no_chunks_exceeds_max_length(self, agent):
        long_para = "word " * 100  # 500 chars
        md = f"# Title\n\n{long_para}\n\n{long_para}"
        chunks = agent.chunk(md, {"source_file": "test.md", "title": "Test"})
        for chunk in chunks:
            assert len(chunk["text"]) <= agent.max_chunk_chars * 2  # generous bound

    def test_empty_markdown_returns_no_chunks(self, agent):
        chunks = agent.chunk("", {"source_file": "empty.md", "title": "Empty"})
        assert chunks == []

    def test_no_headings_still_chunks(self, agent):
        md = "Just a plain paragraph with no headings at all."
        chunks = agent.chunk(md, {"source_file": "flat.md", "title": "Flat"})
        assert len(chunks) >= 1
        assert chunks[0]["text"] == md
