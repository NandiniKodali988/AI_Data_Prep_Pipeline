---
title: DocPipe
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
app_file: streamlit_app.py
pinned: false
---

# DocPipe

**[HuggingFace Space](https://huggingface.co/spaces/NandiniKodali/docpipe)**

A local RAG pipeline that ingests documents in various formats, converts them to clean Markdown, indexes the content into ChromaDB, and lets you have a multi-turn conversation against the full corpus. Images and diagrams get described by Claude Vision so they are searchable too.

## What it does

Drop a file into the Upload tab. DocPipe extracts the content, chunks it, stores it in a local vector database, and generates a summary so you know what was indexed. Then switch to the Chat tab and ask questions — follow-up questions work because the full conversation history is passed back to Claude on each turn.

```
document > format detection > content extraction > chunking > ChromaDB > chat Q&A
```

## Supported formats

| Format | What gets extracted |
|--------|-------------------|
| PDF | text, tables (pdfplumber), embedded images (Claude Vision) |
| DOCX | headings, body text, tables, embedded images |
| PPTX | slide text, speaker notes, images per slide |
| XLSX | each sheet as a Markdown table |
| TXT / MD / RST | passthrough |
| JSON / YAML | wrapped in a code block |
| Images (PNG, JPG, etc.) | described by Claude Vision |

## Setup

```bash
git clone <repo-url>
cd AI_Data_Prep_Pipeline

pip install -r requirements.txt

cp .env.example .env
# add your Anthropic API key to .env
```

## Usage

**Streamlit app (recommended)**

```bash
streamlit run streamlit_app.py
```

Upload a file in the Upload tab — a summary appears automatically after indexing. Switch to the Chat tab to ask questions. Use the "Clear conversation" button in the sidebar to start a new session.

**CLI**

```bash
# index a folder of documents
python main.py --input ./data --output ./output

# ask a question (retrieves chunks and generates a cited answer)
python main.py --query "What is chain-of-thought prompting?"

# raw semantic search (returns chunks without generation)
python main.py --search "time series forecasting" --top-k 5
```

## How it works

Each file goes through a chain of agents:

- **FormatDetectionAgent**: magic bytes first, extension as fallback for zip-based formats (docx/pptx/xlsx all share the same header)
- **PDFAgent / DocxAgent / PptxAgent / XlsxAgent**: format-specific extraction
- **ImageProcessingAgent**: sends embedded images to Claude Vision and gets back a searchable description
- **ChunkingAgent**: splits on headings first, then paragraphs, with a small overlap window so answers that straddle chunk boundaries do not get missed
- **IndexingAgent**: upserts into ChromaDB using a SHA256 ID derived from filename and chunk index, so re-indexing is safe
- **RAGAgent**: formats retrieved chunks as numbered context, sends to Claude with the full conversation history and a strict grounding prompt, returns the answer with inline `[1]`, `[2]` citations

## Evaluation

Evaluated against a 26-question set across 4 documents (2 PDFs, 1 research paper, 1 PPTX):

| Metric | Score |
|--------|-------|
| Recall@1 | 84.6% |
| Recall@3 | 88.5% |
| Recall@5 | 92.3% |
| Precision@3 | 0.782 |
| MRR | 0.867 |

```bash
python eval/evaluate.py --top-k 5
```

## Project structure

```
src/
  agents/
    format_detection_agent.py  # magic bytes + extension fallback
    pdf_agent.py               # PyMuPDF text + pdfplumber tables + images
    office_agent.py            # DocxAgent, PptxAgent, XlsxAgent
    image_processing_agent.py  # Claude Vision descriptions
    chunking_agent.py          # heading and paragraph splits with overlap
    indexing_agent.py          # ChromaDB upsert and search
    rag_agent.py               # multi-turn Q&A and summarization
    metadata_agent.py          # normalizes metadata across formats
    text_agent.py              # plain text and Markdown passthrough
    structured_data_agent.py   # JSON and YAML to Markdown
  pipeline.py                  # orchestrator
streamlit_app.py               # browser UI (Upload + Chat tabs)
main.py                        # CLI
eval/
  evaluate.py                  # Recall@k, Precision@k, MRR
  eval_set.yaml                # 26 questions across 4 documents
tests/                         # 58 tests, all passing
  test_format_detection.py
  test_text_agent.py
  test_chunking_agent.py
  test_indexing_agent.py
  test_office_agent.py
  test_rag_agent.py
pyproject.toml                 # ruff config and pytest settings
packages.txt                   # HuggingFace system dependencies
Dockerfile                     # HuggingFace Spaces deployment
```

## Stack

- Claude (`claude-sonnet-4-6`) — Vision, summarization, and chat Q&A
- ChromaDB — local persistent vector store
- PyMuPDF + pdfplumber — PDF text and table extraction
- python-docx / python-pptx / openpyxl — Office format parsing
- Streamlit — UI
