import re


class ChunkingAgent:
    def __init__(self, max_chunk_chars: int = 2000, overlap_chars: int = 200):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, markdown: str, metadata: dict) -> list[dict]:
        chunks = []
        for heading, body in self._split_by_headings(markdown):
            section_text = f"{heading}\n\n{body}".strip() if heading else body.strip()
            if not section_text:
                continue

            sub_chunks = (
                [section_text]
                if len(section_text) <= self.max_chunk_chars
                else self._split_by_paragraphs(section_text)
            )

            for i, text in enumerate(sub_chunks):
                m = dict(metadata)
                m["chunk_index"] = len(chunks)
                m["section_heading"] = heading.strip() if heading else ""
                m["sub_chunk_index"] = i
                chunks.append({"text": text, "metadata": m})

        return chunks

    def _split_by_headings(self, markdown: str) -> list[tuple[str, str]]:
        pattern = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
        matches = list(pattern.finditer(markdown))

        if not matches:
            return [("", markdown)]

        sections = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            sections.append((match.group(0), markdown[start:end].strip()))

        # content before the first heading (title, abstract, etc.) gets its own unnamed section
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, ("", preamble))

        return sections

    def _split_by_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\n+", text)

        # individual paragraphs that are still too long get word-split
        flat = []
        for p in paragraphs:
            flat.extend(self._split_by_words(p) if len(p) > self.max_chunk_chars else [p])

        # carry a tail of the previous chunk into the next one so retrieval
        # doesn't miss answers that straddle a chunk boundary
        chunks, current, overlap_buf = [], "", ""
        for para in flat:
            candidate = f"{overlap_buf}\n\n{para}".strip() if overlap_buf else para
            if len(current) + len(para) + 2 > self.max_chunk_chars and current:
                chunks.append(current.strip())
                overlap_buf = current[-self.overlap_chars:].strip()
                current = para
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        return chunks or [text]

    def _split_by_words(self, text: str) -> list[str]:
        words = text.split()
        chunks, current_words, current_len = [], [], 0

        for word in words:
            if current_len + len(word) + 1 > self.max_chunk_chars and current_words:
                chunks.append(" ".join(current_words))
                # build overlap by walking backwards through the words we just flushed
                overlap_words, overlap_len = [], 0
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

        return chunks or [text]
