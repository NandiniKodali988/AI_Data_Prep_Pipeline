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
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def answer(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> dict[str, str | list[str] | int]:
        """Generate a grounded answer from retrieved chunks.

        history is a list of prior {"role": ..., "content": ...} turns so the
        model can handle follow-up questions that reference earlier answers.

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

        # prepend prior conversation turns so follow-ups ("tell me more", "how does
        # that compare to X?") have the context they need
        messages = list(history) if history else []
        messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
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

    def rewrite_query(self, question: str, history: list[dict] | None = None) -> str:
        """Rewrite the user's question into a keyword-rich search query.

        Resolves coreferences from conversation history so follow-up questions
        like "tell me more about that" retrieve the right chunks.
        """
        if not history:
            prompt = (
                "Rewrite the following question as a concise, keyword-rich search query "
                "optimized for retrieving relevant document chunks from a vector database. "
                "Return only the rewritten query, no explanation.\n\n"
                f"Question: {question}"
            )
        else:
            # include recent turns so pronouns and references can be resolved
            recent = history[-4:]
            convo = "\n".join(f"{m['role'].capitalize()}: {m['content'][:200]}" for m in recent)
            prompt = (
                "Given the conversation below, rewrite the latest question as a standalone, "
                "keyword-rich search query that resolves any pronouns or references to prior answers. "
                "Return only the rewritten query, no explanation.\n\n"
                f"Conversation:\n{convo}\n\n"
                f"Latest question: {question}"
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = response.content[0].text.strip()
        logger.debug("query rewritten: %r -> %r", question, rewritten)
        return rewritten

    def summarize(self, markdown: str, filename: str) -> str:
        """Return a short bullet-point summary of a document's content."""
        # cap at 4000 chars so we don't blow the token budget on large docs
        content = markdown[:4000]
        prompt = (
            f"Summarize the key points of '{filename}' in 3-5 concise bullet points. "
            f"Base your summary only on the content below.\n\n{content}"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
