"""Chunking Agent — splits Markdown documents into retrieval-ready chunks."""
import re


class ChunkingAgent:
    """
    Splits a Markdown document into chunks that respect structure.

    Strategy:
    1. Split on Markdown headings (#, ##, ###) to keep sections together.
    2. If a section is too long, split further on paragraph boundaries.
    3. Add a 1-sentence overlap between adjacent chunks for context continuity.
    4. Attach metadata (source, chunk index, heading) to every chunk.
    """

    def __init__(self, max_chunk_chars: int = 2000, overlap_chars: int = 200):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, markdown: str, metadata: dict) -> list[dict]:
        """
        Split Markdown into chunks with metadata.

        Returns:
            List of dicts: {"text": str, "metadata": dict}
        """
        sections = self._split_by_headings(markdown)
        chunks = []

        for heading, body in sections:
            section_text = f"{heading}\n\n{body}".strip() if heading else body.strip()
            if not section_text:
                continue

            if len(section_text) <= self.max_chunk_chars:
                sub_chunks = [section_text]
            else:
                sub_chunks = self._split_by_paragraphs(section_text)

            for i, chunk_text in enumerate(sub_chunks):
                chunk_meta = dict(metadata)
                chunk_meta["chunk_index"] = len(chunks)
                chunk_meta["section_heading"] = heading.strip() if heading else ""
                chunk_meta["sub_chunk_index"] = i
                chunks.append({"text": chunk_text, "metadata": chunk_meta})

        return chunks

    def _split_by_headings(self, markdown: str) -> list[tuple[str, str]]:
        """Split markdown into (heading, body) pairs."""
        heading_pattern = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(markdown))

        if not matches:
            return [("", markdown)]

        sections = []
        for i, match in enumerate(matches):
            heading = match.group(0)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            body = markdown[start:end].strip()
            sections.append((heading, body))

        # Content before the first heading
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, ("", preamble))

        return sections

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text on double newlines, adding overlap between chunks."""
        paragraphs = re.split(r"\n\n+", text)
        # Further split any paragraph that is itself too long
        flat_paras = []
        for para in paragraphs:
            if len(para) > self.max_chunk_chars:
                flat_paras.extend(self._split_by_words(para))
            else:
                flat_paras.append(para)

        chunks = []
        current = ""
        overlap_buffer = ""

        for para in flat_paras:
            candidate = (overlap_buffer + "\n\n" + para).strip() if overlap_buffer else para
            if len(current) + len(para) + 2 > self.max_chunk_chars and current:
                chunks.append(current.strip())
                # Keep tail of current chunk as overlap for next
                overlap_buffer = current[-self.overlap_chars :].strip()
                current = para
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    def _split_by_words(self, text: str) -> list[str]:
        """Split a single long paragraph into word-boundary chunks."""
        words = text.split()
        chunks = []
        current_words = []
        current_len = 0

        for word in words:
            # +1 for the space
            if current_len + len(word) + 1 > self.max_chunk_chars and current_words:
                chunks.append(" ".join(current_words))
                # Carry over overlap
                overlap_words = []
                overlap_len = 0
                for w in reversed(current_words):
                    if overlap_len + len(w) + 1 > self.overlap_chars:
                        break
                    overlap_words.insert(0, w)
                    overlap_len += len(w) + 1
                current_words = overlap_words + [word]
                current_len = overlap_len + len(word) + 1
            else:
                current_words.append(word)
                current_len += len(word) + 1

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks if chunks else [text]
