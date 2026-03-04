import logging
import os

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"

# strict grounding — we don't want Claude using its parametric knowledge here,
# only what came back from the vector search
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided "
    "document excerpts. Each excerpt is labelled with a [number] and its source filename. "
    "Cite sources inline using [number] immediately after the relevant statement. "
    "If multiple excerpts support the same point, cite all of them. "
    "If the answer cannot be found in the excerpts, say: "
    "'I could not find an answer in the indexed documents.' "
    "Do not speculate or use knowledge outside the provided context."
)


class RAGAgent:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def answer(self, question: str, chunks: list[dict]) -> dict:
        """Generate a grounded answer from retrieved chunks.

        Returns:
            {
                "answer": str,          # Claude's response with inline citations
                "sources": list[str],   # unique filenames of chunks used
                "chunks_used": int,
            }
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            fname = meta.get("filename", meta.get("source_file", "unknown"))
            section = meta.get("section_heading", "")
            header = f"[{i}] {fname}" + (f" — {section}" if section else "")
            context_parts.append(f"{header}\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)
        user_message = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = response.content[0].text

        # collect unique source filenames in the order they were retrieved
        seen: set[str] = set()
        sources: list[str] = []
        for chunk in chunks:
            meta = chunk["metadata"]
            fname = meta.get("filename", meta.get("source_file", "unknown"))
            if fname not in seen:
                seen.add(fname)
                sources.append(fname)

        logger.debug("RAG answer generated using %d chunks from %s", len(chunks), sources)
        return {"answer": answer_text, "sources": sources, "chunks_used": len(chunks)}
