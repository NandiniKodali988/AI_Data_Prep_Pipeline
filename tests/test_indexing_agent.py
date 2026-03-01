"""Tests for IndexingAgent (uses a temp ChromaDB dir)."""
import tempfile

import pytest

from src.agents.indexing_agent import IndexingAgent


@pytest.fixture
def agent(tmp_path):
    return IndexingAgent(chroma_db_path=str(tmp_path / "chroma_test"))


SAMPLE_CHUNKS = [
    {
        "text": "ChromaDB is a vector database for AI applications.",
        "metadata": {
            "source_file": "/docs/chroma.pdf",
            "title": "ChromaDB Docs",
            "chunk_index": 0,
            "section_heading": "# Introduction",
            "file_type": "pdf",
        },
    },
    {
        "text": "Embeddings represent text as dense vectors in high-dimensional space.",
        "metadata": {
            "source_file": "/docs/chroma.pdf",
            "title": "ChromaDB Docs",
            "chunk_index": 1,
            "section_heading": "## Embeddings",
            "file_type": "pdf",
        },
    },
]


class TestIndexingAgent:
    def test_index_returns_count(self, agent):
        n = agent.index(SAMPLE_CHUNKS)
        assert n == len(SAMPLE_CHUNKS)

    def test_collection_size_matches(self, agent):
        agent.index(SAMPLE_CHUNKS)
        assert agent.collection_size() == len(SAMPLE_CHUNKS)

    def test_upsert_is_idempotent(self, agent):
        agent.index(SAMPLE_CHUNKS)
        agent.index(SAMPLE_CHUNKS)  # second run
        assert agent.collection_size() == len(SAMPLE_CHUNKS)  # no duplicates

    def test_search_returns_results(self, agent):
        agent.index(SAMPLE_CHUNKS)
        results = agent.search("vector database", top_k=2)
        assert len(results) > 0
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_empty_index_returns_no_chunks(self, agent):
        n = agent.index([])
        assert n == 0

    def test_metadata_preserved_after_index(self, agent):
        agent.index(SAMPLE_CHUNKS)
        results = agent.search("ChromaDB vector", top_k=1)
        assert results[0]["metadata"]["title"] == "ChromaDB Docs"
        assert results[0]["metadata"]["file_type"] == "pdf"
