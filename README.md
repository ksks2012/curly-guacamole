# curly-guacamole

A local RAG (Retrieval-Augmented Generation) pipeline built with LangChain, Chroma, and any OpenAI-compatible server (e.g. llama.cpp).

Designed to go from "just works" → **accurate + extensible + maintainable**.

## Features

- **Multi-source ingestion** — PDF, Markdown, plain-text, and **Notion** pages
- **Notion sync** — incremental sync via data source query + Markdown endpoint; content-hash-based change detection skips unchanged pages
- **Web-based upload** — upload and index files directly from the dashboard
- **MMR retrieval** to balance relevance and diversity
- **Hybrid search** — vector similarity fused with BM25 via Reciprocal Rank Fusion (RRF)
- **Re-ranking** (cross-encoder or LLM-based, swappable via config)
- **Query expansion** — generates alternative phrasings to broaden recall
- **Citation-grounded answers** — LLM is forced to cite `[page N, filename]`
- **Incremental indexing** via LangChain `index()` + SQLite record manager (no duplicates on re-run)
- **Raw storage layer** — SQLAlchemy Core / SQLite-backed store for pages and blocks, enabling re-embedding without re-fetching from Notion
- **Document-level filtering** — scope search to a specific `doc_id` from the dashboard

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

# --- Upload ---
upload_dir: "./data/uploads"      # directory where uploaded files are saved

# --- Vector store ---
persist_directory: "./my_db"
setup_rag_collection: "rag_collection"
db_url: "sqlite:///my_db/record_manager_cache.sql"
batch_limit: 256          # chunk count threshold for switching indexing strategy

# --- Raw storage (Notion sync) ---
raw_db_path: "./my_db/raw.db"

# --- Notion ---
# notion_token: "secret_..."          # Notion integration secret
# notion_workspace_id: "My Workspace" # logical workspace name (free string)
# notion_database_id:  "My Database"  # Notion database title or UUID

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

### Notion sync (programmatic)

```python
from rag.knowledge.models import Workspace
from rag.knowledge.store import RawStore
from rag.ingest.notion.client import NotionClient
from rag.ingest.notion.sync import NotionSyncPipeline

# 1. Resolve the data_source_id from your Notion database
client = NotionClient(token="secret_...")
db = client.get_database("your-database-uuid")
data_source_id = db["data_sources"][0]["id"]

# 2. Set up storage
store = RawStore("./my_db/raw.db")
workspace = Workspace.new("My Workspace")

# 3. Run sync
pipeline = NotionSyncPipeline(
    token="secret_...",
    workspace=workspace,
    store=store,
    data_source_id=data_source_id,   # omit to fall back to /v1/search
    full_sync=False,                  # True to ignore stored cursor
)
result = pipeline.sync()
print(result.as_dict())
# {"pages_seen": 42, "pages_updated": 3, "pages_skipped": 39, "errors": 0}
```

The pipeline stores each page's Markdown content (fetched from `/v1/pages/{id}/markdown`) in the raw SQLite store. Subsequent runs skip pages whose content hash has not changed.

### Debug dashboard

```bash
python cmd/dashboard.py
# open http://localhost:8888
```

The dashboard is an engineering tool for tuning retrieval — not a demo. It has two tabs:

**Search tab**

- Enter a query, choose `top-k` and `fetch-k`, optionally enable `Rerank`
- Results show relevance score, page, chunk ID, filename, and content preview
- Click any result card to inspect full metadata and chunk content in the right panel
- Enable **Filter by doc** to scope the search to a single `doc_id`

With rerank on:
- Left column: all `fetch-k` vector candidates
- Right column: top-`k` reranked results with rank-change indicators (`▲N` / `▼N`)
- Blue left border = chunk that survived reranking; green border = reranked result
- Cross-encoder score displayed alongside vector relevance score for comparison

**Index tab**

- Set chunk size, overlap, and an optional `doc-id` override
- Drop a PDF or click to select — the file is saved to `upload_dir` immediately and embedding runs in the background (upload confirmation appears before embedding completes)
- Indexed document list updates automatically after each upload
- The Search tab's filter dropdown is refreshed with any newly indexed `doc_id`

### Indexing multiple documents with isolation (programmatic)

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
│   └── dashboard.py              # NiceGUI debug dashboard entry point (bootstrap + tab assembly)
├── ui/
│   ├── search_tab.py             # Search tab UI — query bar, filters, result columns, chunk detail
│   ├── index_tab.py              # Index tab UI — chunking options, file upload, doc list
│   ├── search_controller.py      # Search tab logic (state, search, filter)
│   └── index_controller.py       # Index tab logic (save file, embed, list docs)
├── etc/
│   └── config.yaml               # All runtime settings
├── rag/
│   ├── client.py                 # LocalLlamaClient — coordinates all RAG components
│   ├── engine.py                 # RAGEngine — query expansion, retrieval, rerank, generation
│   ├── indexer.py                # Indexer — SQLRecordManager lifecycle and document ingestion
│   ├── prompt.py                 # RAG_PROMPT, QUERY_EXPANSION_PROMPT
│   ├── reranker.py               # BaseReranker, CrossEncoderReranker, LLMReranker, RerankerFactory
│   ├── ingest/
│   │   ├── chunker.py            # Chunking strategies: recursive / heading-aware / semantic
│   │   ├── document_ingester.py  # Unified ingestion entry point (PDF, MD, TXT)
│   │   ├── schema.py             # Canonical chunk metadata schema
│   │   ├── strategies.py         # ChunkStrategy enum + auto-selection logic
│   │   ├── notion/
│   │   │   ├── client.py         # NotionClient — REST API wrapper (auth, pagination, retry)
│   │   │   ├── sync.py           # NotionSyncPipeline — incremental page sync to RawStore
│   │   │   ├── chunker.py        # NotionChunker — structure-aware heading/section chunker
│   │   │   ├── embedder.py       # NotionEmbedder — RawStore → Chroma embedding pipeline
│   │   │   └── pipeline.py       # NotionRAGClient — unified sync + embed + query entry point
│   │   └── parsers/
│   │       ├── pdf.py            # PDF → Documents (page-level)
│   │       ├── markdown.py       # Markdown → Documents (heading-section-level)
│   │       └── text.py           # Plain-text → Documents
│   ├── knowledge/
│   │   ├── models.py             # Domain models: Workspace, Page, Block, Chunk, DocumentVersion
│   │   ├── metadata.py           # ChunkMetadata — canonical Chroma metadata schema
│   │   └── store.py              # RawStore — SQLAlchemy Core / SQLite raw storage layer
│   └── retrieval/
│       ├── bm25.py               # BM25Index + RRF fusion
│       └── filters.py            # SearchFilter — Chroma where-clause builder
├── utils/
│   ├── config.py                 # AppConfig — typed properties over config.yaml
│   ├── file_processor.py         # PDF chunking, YAML / JSON / CSV helpers
│   └── logger.py                 # AppLogger — centralised logging setup
├── data/
│   └── uploads/                  # Uploaded files (path configurable via upload_dir)
├── my_db/
│   ├── chroma.sqlite3            # Chroma vector store (auto-created)
│   ├── record_manager_cache.sql  # LangChain record manager
│   └── raw.db                    # Raw storage layer for Notion sync (auto-created)
├── main.py                       # Entry point
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
  +
BM25 keyword search
  │
RRF fusion + de-duplicate by chunk_id
  │
  ▼
reranker  (cross-encoder or LLM → top k)
  │
  ▼
citation-grounded LLM answer  [page N, filename]
```

## Notion Sync Pipeline

```
NotionClient.get_database(database_id)
  │  resolves data_source_id from db["data_sources"]
  ▼
NotionClient.iter_data_source_pages(data_source_id)
  │  POST /v1/data_sources/{id}/query  (paginated, cursor-resumable)
  │  falls back to iter_all_pages() if data_source_id not provided
  ▼
for each page:
  NotionClient.get_page_markdown(page_id)
    │  GET /v1/pages/{id}/markdown
    ▼
  SHA-256(markdown) == stored hash?
    yes → skip
    no  → upsert Page + Block(LOCAL_PAGE_TEXT) + DocumentVersion → RawStore
  │
  cursor saved after every batch (supports partial-run resume)
```

