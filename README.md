# curly-guacamole

A local RAG (Retrieval-Augmented Generation) pipeline built with LangChain, Chroma, and any OpenAI-compatible server (e.g. llama.cpp).

Designed to go from "just works" → **accurate + extensible + maintainable**.

## Features

- **PDF ingestion** with deterministic chunking and enriched metadata
- **MMR retrieval** to balance relevance and diversity
- **Re-ranking** (cross-encoder or LLM-based, swappable via config)
- **Query expansion** — generates alternative phrasings to broaden recall
- **Citation-grounded answers** — LLM is forced to cite `[page N, filename]`
- **Incremental indexing** via LangChain `index()` + SQLite record manager (no duplicates on re-run)
- **Document-level filtering** with `doc_id` metadata
- **Debug dashboard** — NiceGUI web UI for tuning retrieval, visualising rerank diff, and inspecting chunk metadata

## Requirements

- Python 3.10+
- A running OpenAI-compatible server for embeddings and/or LLM inference (e.g. [llama.cpp server](https://github.com/ggerganov/llama.cpp))

## Installation

```bash
# Activate your virtual environment first
source ../rt-sandbox/bin/activate

# Install the package and its dependencies
pip install .
```

## Configuration

Edit `etc/config.yaml`:

```yaml
# --- Servers ---
embed_base: "http://localhost:8080/v1/"
llm_base:   "http://localhost:8080/v1/"
embed_model: "text-embedding-ada-002"
llm_model:   "local-model"
api_key:     "sk-no-key-required"
llm_kwargs:  {}

# --- Vector store ---
persist_directory: "./my_db"
setup_rag_collection: "rag_collection"
db_url: "sqlite:///my_db/record_manager_cache.sql"
batch_limit: 256          # chunk count threshold for switching indexing strategy

# --- Reranker ---
reranker_type:  "cross_encoder"   # "cross_encoder" | "llm" | "none"
reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Query expansion ---
query_expansion_enabled: false
query_expansion_n: 3              # number of extra phrasings to generate

# --- Logging ---
log_level:   "INFO"               # DEBUG | INFO | WARNING | ERROR
log_format:  "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
log_datefmt: "%H:%M:%S"

# --- Runtime / testing ---
pdf_path:   "/path/to/your/document.pdf"
test_query: "What are the main contents of this document?"
test_search: "your keyword"
```

## Usage

### RAG pipeline (CLI)

```bash
python main.py
```

`main.py` will answer the question in `test_query` using the indexed Chroma collection.
To index a PDF first, uncomment `client.add_pdf(config.pdf_path)` in `main.py`.

### Debug dashboard

```bash
python cmd/dashboard.py
# open http://localhost:8888
```

The dashboard is an engineering tool for tuning retrieval — not a demo.

**Phase 1 — Vector search**
- Enter a query, choose `top-k` and `fetch-k`
- Results show relevance score, page, chunk ID, filename, and content preview
- Click any result card to inspect full metadata and chunk content in the right panel

**Phase 2 — Rerank visualisation** (enable the `Rerank` checkbox)
- Left column: all `fetch-k` vector candidates
- Right column: top-`k` reranked results with rank-change indicators (`▲N` / `▼N`)
- Blue left border = chunk that survived reranking; green = reranked result
- Cross-encoder score displayed alongside vector relevance score for comparison

### Indexing multiple documents with isolation

```python
client.add_pdf("report_2024.pdf", doc_id="report_2024")
client.add_pdf("manual.pdf",      doc_id="manual")

# Query scoped to one document
resp = client.answer_query("What changed in 2024?", doc_id="report_2024")

# Query across all documents
resp = client.answer_query("Summarise both documents")
```

### Enabling query expansion at call time

```python
resp = client.answer_query("What is the methodology?", expand_query=True)
```

## Project Structure

```
.
├── cmd/
│   └── dashboard.py          # NiceGUI debug dashboard (Phase 1+2)
├── etc/
│   └── config.yaml           # All runtime settings
├── rag/
│   ├── __init__.py
│   ├── client.py             # LocalLlamaClient — retrieval, reranking, generation
│   ├── prompt.py             # RAG_PROMPT, QUERY_EXPANSION_PROMPT
│   └── reranker.py           # BaseReranker, CrossEncoderReranker, LLMReranker
├── utils/
│   ├── __init__.py
│   ├── config.py             # AppConfig — typed properties over config.yaml
│   ├── file_processor.py     # PDF chunking, YAML / JSON / CSV helpers
│   └── logger.py             # AppLogger — centralised logging setup
├── my_db/                    # Chroma vector store (auto-created)
├── main.py                   # Entry point
├── setup.py
└── requirements.txt
```

## RAG Pipeline

```
query
  │
  ├─[query expansion]─► N alternative phrasings (optional, via LLM)
  │
  ▼
MMR vector search  (fetch_k candidates per phrasing)
  │
de-duplicate by chunk_id
  │
  ▼
reranker  (cross-encoder or LLM → top k)
  │
  ▼
citation-grounded LLM answer  [page N, filename]
```

