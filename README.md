# DocPipe

A local RAG pipeline that ingests documents in various formats, converts them to clean Markdown, indexes the content into ChromaDB, and lets you ask questions against the full corpus. Images and diagrams get described by Claude Vision so they're searchable too.

## What it does

Drop a folder of PDFs, Word docs, PowerPoints, spreadsheets, or plain text files into the pipeline. It extracts the content, chunks it intelligently, and stores it in a local vector database. Then you can ask natural language questions and get grounded answers with citations pointing back to the source documents.

```
document → format detection → content extraction → chunking → ChromaDB → RAG Q&A
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
cd AI_Data_Prep_Piepline

pip install -r requirements.txt

cp .env.example .env
# add your Anthropic API key to .env
```

## Usage

**Streamlit app (recommended)**

```bash
streamlit run app.py
```

Upload files through the browser, then switch to the Ask tab to query them.

**CLI**

```bash
# index a folder of documents
python main.py --input ./data --output ./output

# ask a question (retrieves chunks + generates a cited answer)
python main.py --query "What is chain-of-thought prompting?"

# raw semantic search (returns chunks without generation)
python main.py --search "time series forecasting" --top-k 5
```

## How it works

Each file goes through a chain of agents:

- **FormatDetectionAgent** — magic bytes first, extension as fallback for zip-based formats (docx/pptx/xlsx all share the same header)
- **PDFAgent / DocxAgent / PptxAgent / XlsxAgent** — format-specific extraction
- **ImageProcessingAgent** — sends embedded images to Claude Vision and gets back a searchable description
- **ChunkingAgent** — splits on headings first, then paragraphs, with a small overlap window so answers that straddle chunk boundaries don't get missed
- **IndexingAgent** — upserts into ChromaDB using a SHA256 ID derived from filename + chunk index, so re-indexing is safe
- **RAGAgent** — formats retrieved chunks as numbered context, sends to Claude with a strict grounding prompt, returns the answer with inline `[1]`, `[2]` citations

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
    format_detection_agent.py
    pdf_agent.py
    office_agent.py        # DocxAgent, PptxAgent, XlsxAgent
    image_processing_agent.py
    chunking_agent.py
    indexing_agent.py
    rag_agent.py
    metadata_agent.py
    text_agent.py
    structured_data_agent.py
  pipeline.py
app.py                     # Streamlit UI
main.py                    # CLI
eval/
  evaluate.py
  eval_set.yaml
tests/                     # 45 tests, all passing
```

## Stack

- Claude (`claude-sonnet-4-6`) — Vision + RAG generation
- ChromaDB — local persistent vector store
- PyMuPDF + pdfplumber — PDF text and table extraction
- python-docx / python-pptx / openpyxl — Office format parsing
- Streamlit — UI
