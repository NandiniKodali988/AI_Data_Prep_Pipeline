"""Tests for RAGAgent — Anthropic API calls are mocked."""
from unittest.mock import MagicMock, patch

import pytest

from src.agents.rag_agent import RAGAgent


def make_chunk(text: str, filename: str, section: str = "") -> dict:
    return {
        "text": text,
        "metadata": {
            "source_file": f"/docs/{filename}",
            "filename": filename,
            "section_heading": section,
            "chunk_index": 0,
        },
        "distance": 0.1,
    }


def mock_response(answer_text: str) -> MagicMock:
    """Build a fake Anthropic messages.create() response."""
    content_block = MagicMock()
    content_block.text = answer_text
    response = MagicMock()
    response.content = [content_block]
    return response


@pytest.fixture
def agent():
    with patch("src.agents.rag_agent.anthropic.Anthropic"):
        a = RAGAgent()
        a.client = MagicMock()
        return a


class TestRAGAgent:
    def test_answer_returns_expected_keys(self, agent):
        agent.client.messages.create.return_value = mock_response("Paris [1].")
        chunks = [make_chunk("Paris is the capital of France.", "geo.pdf")]
        result = agent.answer("What is the capital of France?", chunks)
        assert "answer" in result
        assert "sources" in result
        assert "chunks_used" in result

    def test_answer_text_matches_mock(self, agent):
        agent.client.messages.create.return_value = mock_response("The answer is 42 [1].")
        chunks = [make_chunk("The answer is 42.", "trivia.pdf")]
        result = agent.answer("What is the answer?", chunks)
        assert result["answer"] == "The answer is 42 [1]."

    def test_sources_deduped_in_retrieval_order(self, agent):
        agent.client.messages.create.return_value = mock_response("Some answer [1][2].")
        chunks = [
            make_chunk("First chunk from doc A.", "a.pdf"),
            make_chunk("Second chunk from doc A.", "a.pdf"),
            make_chunk("First chunk from doc B.", "b.pdf"),
        ]
        result = agent.answer("anything", chunks)
        # a.pdf appears twice in chunks but should only appear once in sources
        assert result["sources"] == ["a.pdf", "b.pdf"]

    def test_chunks_used_count(self, agent):
        agent.client.messages.create.return_value = mock_response("Answer.")
        chunks = [make_chunk(f"chunk {i}", "doc.pdf") for i in range(4)]
        result = agent.answer("question", chunks)
        assert result["chunks_used"] == 4

    def test_context_includes_all_chunks(self, agent):
        agent.client.messages.create.return_value = mock_response("ok")
        chunks = [
            make_chunk("Alpha content.", "alpha.pdf", "Introduction"),
            make_chunk("Beta content.", "beta.pdf"),
        ]
        agent.answer("question", chunks)

        call_args = agent.client.messages.create.call_args
        user_message = call_args.kwargs["messages"][0]["content"]
        # both filenames should appear in the context sent to Claude
        assert "alpha.pdf" in user_message
        assert "beta.pdf" in user_message

    def test_section_heading_included_in_context(self, agent):
        agent.client.messages.create.return_value = mock_response("ok")
        chunks = [make_chunk("Some content.", "report.pdf", "Key Findings")]
        agent.answer("question", chunks)

        call_args = agent.client.messages.create.call_args
        user_message = call_args.kwargs["messages"][0]["content"]
        assert "Key Findings" in user_message

    def test_empty_chunks_still_calls_api(self, agent):
        # RAGAgent should still call the API even with no chunks
        # (the caller is responsible for not calling with empty results)
        agent.client.messages.create.return_value = mock_response("I could not find an answer.")
        result = agent.answer("question", [])
        assert result["chunks_used"] == 0
        agent.client.messages.create.assert_called_once()
