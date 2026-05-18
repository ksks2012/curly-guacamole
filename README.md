# curly-guacamole

A local RAG (Retrieval-Augmented Generation) pipeline built with LangChain, Chroma, and any OpenAI-compatible server (e.g. llama.cpp).

Designed to go from "just works" → **accurate + extensible + maintainable**.

## Features

### Retrieval Foundation
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

### Knowledge Layer (Stage B)
- **Knowledge extraction** (B.1) — LLM-driven per-chunk enrichment: summary, keywords, entities, topics, and auto-generated questions written back into Chroma metadata
- **QA generation** (B.2) — generates question-answer pairs per document; stored in a dedicated QA Chroma collection for semantic question matching at query time
- **Topic clustering** (B.3) — K-Means over chunk embeddings with LLM-generated cluster labels; assigns `topic_*` tags to every chunk
- **Cross-document linking** (B.4) — cosine similarity linking at chunk level (`related_chunks`) and page level (`related_pages`); stored in metadata so retrieval results carry "see also" links at zero query-time cost

### Memory System (Stage C)
- **Conversation memory** (C.1) — persistent short/medium-term memory per session: tracks `active_project`, rolling `current_topics`, and `recent_questions`; automatically injected into every RAG prompt and updated after each answer
- **Topic extraction** — LLM extracts 1-5 topic tags from each Q-A turn to keep `current_topics` up to date
- **Project inference** — periodically infers `active_project` from recent conversation history via LLM
- **SQLite-backed sessions** — all memory persists across restarts; multiple named sessions supported
- **Semantic user memory** (C.2) — long-term recency-weighted interest profile across all sessions; EMA scoring surfaces frequently and recently discussed topics; injected into every RAG prompt as a `Frequent Research Areas` hint
- **Knowledge timeline** (C.3) — daily activity log aggregating topics and retrieved document IDs per calendar day; enables queries like "what was I working on last week?"; injects a `Recent Activity` block into the prompt
- **Research session tracking** (C.4) — named research sessions that group related queries, retrieved documents, and free-form notes; an *active session* auto-accumulates every Q-A turn; session context injected into the RAG prompt; supports archive/restore lifecycle

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

# --- Conversation memory (Stage C.1) ---
memory_db_path:           "./my_db/memory.sqlite"
memory_default_session:   "default"
memory_max_recent:        20      # max Q-A turns kept in recent_questions
memory_max_topics:        10      # max entries in current_topics
memory_extract_topics:    true    # call LLM to extract topic tags after each turn
memory_auto_infer_project: true   # periodically infer active_project from conversation

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

The dashboard is an engineering tool for tuning retrieval — not a demo. It has four tabs:

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

**Trace tab**

- Detailed pipeline trace for a query: expansion phrasings, per-stage candidate lists, reranker scores
- Useful for diagnosing retrieval quality without modifying code

**Index tab**

- Set chunk size, overlap, and an optional `doc-id` override
- Drop a PDF or click to select — the file is saved to `upload_dir` immediately and embedding runs in the background (upload confirmation appears before embedding completes)
- Indexed document list updates automatically after each upload
- The Search tab's filter dropdown is refreshed with any newly indexed `doc_id`

**Config tab**

- Edit all `etc/config.yaml` settings through a form-based UI without restarting
- Fields with immediate effect are marked `● active`; fields requiring a restart are marked `⚠ restart`
- Save writes to disk and hot-reloads applicable settings instantly

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

### Knowledge Layer (Stage B)

```python
# B.1 — enrich all chunks of a document with LLM-extracted metadata
client.enrich_doc("my_doc")

# B.2 — generate Q-A pairs and index them in the QA collection
client.generate_qa("my_doc")
# semantic search over generated questions
results = client.qa_search("How does chunking work?", k=5)

# B.3 — cluster all indexed chunks into topics and label them
map = client.cluster_topics(n_clusters=8)
# map.labels: {chunk_id: "topic_rag"}, map.summary: {"topic_rag": ["id1", ...]}

# B.4 — compute cross-document chunk-level links
client.link_chunks(top_k=5, threshold=0.75)
# compute page-level centroid links
client.link_pages(top_k=5, threshold=0.70)
# read back at retrieval time
related_chunks = client.get_related_chunks("chroma-chunk-id")
related_pages  = client.get_related_pages("my_doc")
```

### Conversation Memory (Stage C.1)

Memory is automatically active — every `answer_query()` call injects the current session
context into the prompt and records the turn afterward.

```python
# Override the active project label (also inferred automatically every 10 turns)
client.set_active_project("Building Notion AI Knowledge System")

# Inspect current session state
state = client.get_memory_state()
print(state.active_project)    # "Building Notion AI Knowledge System"
print(state.current_topics)    # ["RAG Architecture", "Vector Search", ...]
print(state.recent_questions)  # [ConversationTurn(...), ...]

# Switch to a different named session (e.g., per user or per project)
client.switch_memory_session("project-b")

# List all sessions
sessions = client.list_sessions()

# Clear the current session's history
client.clear_memory_session()
```

### Semantic User Memory (Stage C.2)

Builds a long-term interest profile across all sessions using recency-weighted scoring.
Automatically updated after every `answer_query()` turn.

```python
# Inspect the interest profile
interests = client.get_user_interests(n=10)
# [{"topic": "RAG Architecture", "count": 12, "weight": 8.34, ...}, ...]

profile = client.get_user_profile()
print(profile.top_interests)      # top 20 weighted topics
print(profile.total_topics_seen)  # distinct topic count
```

### Knowledge Timeline (Stage C.3)

Logs daily activity automatically. Useful for reviewing what was worked on over time.

```python
# Recent activity
recent = client.get_timeline_recent(days=30)
# [{"date": "2026-05-17", "topics": ["RAG", "Memory"], "question_count": 5}, ...]

# Filter by month
may = client.get_timeline_period(2026, month=5)

# Yearly topic frequency map
summary = client.get_yearly_summary(2026)
# {"RAG Architecture": 34, "Vector Search": 18, ...}
```

### Research Session Tracking (Stage C.4)

Group related queries, retrieved documents, and notes under a named research session.

```python
# Start a new research session (automatically becomes the active session)
session = client.start_research_session("Agentic RAG research", tags=["RAG", "Agents"])

# Every subsequent answer_query() call auto-records the query and doc_ids
client.answer_query("What is the ReAct pattern?")
client.answer_query("How does Reflexion differ from ReAct?")

# Add a note (manually written or LLM-generated)
client.add_research_note(
    "ReAct alternates reasoning and acting; Reflexion adds self-evaluation.",
    source_doc_ids=["doc-42"],
)

# Inspect the session
session = client.get_research_session()
print(session.queries)   # ["What is the ReAct pattern?", ...]
print(session.doc_ids)   # all unique docs retrieved across the session
notes = client.get_research_notes()

# List all active sessions
all_sessions = client.list_research_sessions()

# Archive when done
client.archive_research_session()

# Restore or browse archived sessions
archived = client.list_research_sessions(archived=True)
```

## Project Structure

```
.
├── cmd/
│   └── dashboard.py              # NiceGUI debug dashboard entry point (bootstrap + tab assembly)
├── ui/
│   ├── search_tab.py             # Search tab UI — query bar, filters, result columns, chunk detail
│   ├── trace_tab.py              # Trace tab UI — per-stage pipeline diagnostics
│   ├── index_tab.py              # Index tab UI — chunking options, file upload, doc list
│   ├── config_tab.py             # Config tab UI — form-based schema-driven settings editor
│   ├── search_controller.py      # Search tab logic (state, search, filter)
│   ├── index_controller.py       # Index tab logic (save file, embed, list docs)
│   └── config_controller.py      # Config tab logic (load, save, hot-reload, schema)
├── etc/
│   └── config.yaml               # All runtime settings
├── rag/
│   ├── client.py                 # LocalLlamaClient — coordinates all RAG components
│   ├── engine.py                 # RAGEngine — query expansion, retrieval, rerank, generation, memory
│   ├── indexer.py                # Indexer — SQLRecordManager lifecycle and document ingestion
│   ├── prompt.py                 # All prompt templates (RAG, expansion, knowledge, memory)
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
│   │   ├── store.py              # RawStore — SQLAlchemy Core / SQLite raw storage layer
│   │   ├── extractor.py          # KnowledgeExtractor — LLM-driven per-chunk enrichment (B.1)
│   │   ├── qa_generator.py       # QAGenerator — Q-A pair generation and QA index (B.2)
│   │   ├── clusterer.py          # TopicClusterer — K-Means + LLM label assignment (B.3)
│   │   ├── linker.py             # CrossDocLinker — chunk and page similarity linking (B.4)
│   │   └── manager.py            # KnowledgeManager — coordinates B.1–B.4 operations
│   ├── memory/
│   │   ├── models.py             # ConversationTurn, SessionState dataclasses (C.1)
│   │   ├── store.py              # MemoryStore — SQLite persistence for all memory subsystems (C.1–C.4)
│   │   ├── manager.py            # ConversationMemory — session lifecycle, topic extraction, prompt injection (C.1)
│   │   ├── user_memory.py        # UserMemoryManager — EMA-weighted long-term interest profile (C.2)
│   │   ├── timeline.py           # KnowledgeTimeline — daily activity log with topic + doc tracking (C.3)
│   │   └── research_session.py   # ResearchSessionManager — named sessions, queries, notes lifecycle (C.4)
│   └── retrieval/
│       ├── bm25.py               # BM25Index + RRF fusion
│       └── filters.py            # SearchFilter — Chroma where-clause builder
├── utils/
│   ├── config.py                 # AppConfig — typed properties over config.yaml
│   ├── file_processor.py         # PDF chunking, YAML / JSON / CSV helpers
│   └── logger.py                 # AppLogger — centralised logging setup
├── testing/
│   ├── testing_extractor.py      # B.1 knowledge extraction unit tests
│   ├── testing_qa.py             # B.2 QA generation unit tests
│   ├── testing_clusterer.py      # B.3 topic clustering unit tests
│   ├── testing_memory.py         # C.1 conversation memory unit tests
│   ├── testing_user_memory.py    # C.2/C.3 user memory and timeline unit tests
│   ├── testing_research_session.py # C.4 research session unit tests
│   └── ...                       # additional integration and pipeline tests
├── data/
│   └── uploads/                  # Uploaded files (path configurable via upload_dir)
├── my_db/
│   ├── chroma.sqlite3            # Chroma vector store (auto-created)
│   ├── record_manager_cache.sql  # LangChain record manager
│   ├── raw.db                    # Raw storage layer for Notion sync (auto-created)
│   └── memory.sqlite             # Conversation memory store (auto-created)
├── main.py                       # Entry point
├── setup.py
└── requirements.txt
```

## RAG Pipeline

```
query
  │
  ├─[conversation memory]─► inject active_project + current_topics + recent Q-A into prompt
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
  │
  └─[conversation memory]─► save turn, extract topics, update session state
```

## Knowledge Enrichment Pipeline (Stage B)

```
indexed chunks in Chroma
  │
  ├─B.1 enrich_doc()──► LLM extracts summary / keywords / entities / topics / questions
  │                      written back into Chroma metadata
  │
  ├─B.2 generate_qa()─► LLM generates Q-A pairs per chunk
  │                      stored in dedicated QA Chroma collection
  │
  ├─B.3 cluster_topics()► K-Means over embeddings → LLM labels → topic_* tags in metadata
  │
  └─B.4 link_chunks()──► cosine similarity matrix → related_chunk_ids in metadata
       link_pages()────► centroid similarity matrix → related_doc_ids in metadata
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

