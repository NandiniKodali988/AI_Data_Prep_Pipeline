# CLAUDE.md

This file documents the project architecture, conventions, and development workflow for Claude Code.

## Project overview

DocPipe is a local RAG pipeline. It ingests documents (PDF, DOCX, PPTX, XLSX, plain text, images), converts them to clean Markdown, indexes the content into ChromaDB, and answers natural language questions against the corpus. Embedded images are described by Claude Vision so they are searchable.

## Architecture

The pipeline is a chain of single-responsibility agents. Each agent does one thing.

```
file → FormatDetectionAgent → format-specific agent → ImageProcessingAgent (if needed)
     → ChunkingAgent → IndexingAgent → ChromaDB
                                         ↓
question → IndexingAgent.search() → RAGAgent → cited answer
```

### Agents

| Agent | File | Responsibility |
|-------|------|----------------|
| FormatDetectionAgent | `src/agents/format_detection_agent.py` | magic bytes first, extension fallback for zip-based formats |
| PDFAgent | `src/agents/pdf_agent.py` | PyMuPDF text + pdfplumber tables + image extraction |
| DocxAgent | `src/agents/office_agent.py` | iterates XML body in order (paragraphs and tables interleaved) |
| PptxAgent | `src/agents/office_agent.py` | per-slide sections, speaker notes, picture shapes |
| XlsxAgent | `src/agents/office_agent.py` | each sheet as a Markdown table, capped at 200 rows / 20 cols |
| TextAgent | `src/agents/text_agent.py` | .txt/.md/.rst passthrough |
| StructuredDataAgent | `src/agents/structured_data_agent.py` | JSON/YAML wrapped in a code block |
| ImageProcessingAgent | `src/agents/image_processing_agent.py` | base64 → Claude Vision API → description |
| MetadataAgent | `src/agents/metadata_agent.py` | normalizes metadata schema across formats |
| ChunkingAgent | `src/agents/chunking_agent.py` | heading splits → paragraph fallback, small overlap window |
| IndexingAgent | `src/agents/indexing_agent.py` | upserts into ChromaDB, IDs are SHA256(source_file + chunk_index) |
| RAGAgent | `src/agents/rag_agent.py` | formats chunks as numbered context, strict grounding prompt, inline citations |

### Pipeline entry points

- `src/pipeline.py` — orchestrator; `process_single()` for one file (used by Streamlit), `run()` for a directory
- `main.py` — CLI
- `streamlit_app.py` — browser UI

## Running the project

```bash
# install dependencies
pip install -r requirements.txt

# set the Anthropic API key
cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY=sk-...

# process a folder of documents
python main.py --input ./data --output ./output

# ask a question
python main.py --query "What is chain-of-thought prompting?"

# raw semantic search without generation
python main.py --search "time series forecasting" --top-k 5

# browser UI
streamlit run streamlit_app.py
```

## Running tests

```bash
pytest tests/ -v
```

45 tests covering format detection, chunking, indexing, text agent, and office agents. Claude Vision calls are mocked.

## Running evaluation

```bash
python eval/evaluate.py --top-k 5
```

Evaluates against `eval/eval_set.yaml` (26 questions across 4 documents). Reports Recall@1/3/5, Precision@3, and MRR.

## Key design decisions

**Idempotent indexing** — ChromaDB document IDs are SHA256 hashes of `source_file + chunk_index`. Re-indexing the same file is safe and does not create duplicates.

**Ordered DOCX extraction** — `doc.paragraphs` and `doc.tables` are separate lists that lose the original interleaving order. DocxAgent iterates `doc.element.body` directly so tables appear in the right place relative to paragraphs.

**Merged cell deduplication** — merged cells in DOCX share the same underlying `_tc` XML element. DocxAgent deduplicates by `id(cell._tc)` to avoid repeated cell text in table output.

**Lazy Streamlit initialization** — all agents are initialized inside `@st.cache_resource` functions called within the page flow, not at module level. This lets Streamlit serve health checks immediately on startup.

**Strict RAG grounding** — RAGAgent uses a system prompt that instructs the model to answer only from the provided excerpts and say "I don't know" if the answer isn't there. This reduces hallucination.

## After writing Python files

Always run these two commands after editing any Python file:

```bash
ruff check --fix <filename>
ruff format <filename>
```

Or for the whole project:

```bash
ruff check --fix src/ main.py streamlit_app.py
ruff format src/ main.py streamlit_app.py
```

Ruff config is in `pyproject.toml`. Rules: E/W (pycodestyle), F (pyflakes), I (isort), UP (pyupgrade).

## Code conventions

- No type annotations (keeps the code concise for a personal project)
- f-strings for string formatting
- `logger = logging.getLogger(__name__)` in every agent module
- Exceptions are caught per-file in the pipeline; a bad file logs an error and processing continues
- Comments explain *why*, not *what*

## Environment

Requires `ANTHROPIC_API_KEY` in the environment or a `.env` file. The app will start without it but RAG queries and image processing will fail at call time.

## Deployment

Deployed on HuggingFace Spaces (Docker SDK, port 7860). The `chroma_db/` directory is committed to the repo (tracked via Git LFS) so the Space starts with a pre-built index.

Add `ANTHROPIC_API_KEY` as a Space secret under Settings → Variables and secrets.
