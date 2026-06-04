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

### Repository Intelligence Layer (Stage GCR)

#### GCR1 — Repository Foundation
- **Repository scanner** (GCR1.1) — walks directory tree, produces a hash-stamped `RepoManifest`; incremental diff (`ManifestDiff`) identifies added / modified / deleted files between scans
- **AST-aware code parsing** (GCR1.2) — `libcst`-based Python parser extracts `module`, `class`, `function`, and `method` chunks with exact line boundaries and SHA-256 content hashes; qualified names handle nested classes / methods
- **Symbol registry** (GCR1.3) — in-memory `SymbolStore` indexes all named symbols by file, type, visibility (`public` / `private` / `dunder`), and parent; serialisable to JSON for incremental runs
- **Multi-resolution code indexing** (GCR1.4) — four Chroma collections at increasing granularity (`repo`, `file`, `symbol`, `block`); incremental upsert compares `content_hash` to skip unchanged symbols; per-repo pruning removes stale documents
- **Git snapshot system** (GCR1.5) — `GitReader` wraps subprocess git to produce `CommitInfo` and `FileSnapshot` objects without external git library dependencies; `SnapshotStore` accumulates file history for temporal analysis

#### GCR2 — Symbol & Git Knowledge Graph
- **Dependency graph** (GCR2.1) — `libcst`-based `_EdgeCollector` extracts `IMPORTS`, `EXTENDS`, `IMPLEMENTS`, and `CALLS` edges; stored in a SQLite `GraphStore` with idempotent upserts and per-repo / per-file delete helpers
- **Symbol evolution tracking** (GCR2.2) — `build_symbol_evolutions()` processes chronologically ordered `FileSnapshot` objects to produce `SymbolEvolution` records tracking `introduced_in`, `modified_in`, and `deleted_in` commit hashes per symbol
- **Diff semantic analysis** (GCR2.3) — `DiffAnalyzer` sends the unified git diff for a symbol to an LLM and stores a one-line `change_summary` on each `SymbolEvolution`; gracefully degrades to `""` on empty input or LLM error
- **Commit semantic indexing** (GCR2.4) — `CommitAnalyzer` derives `affected_symbols` from evolution records and generates an LLM summary; `CommitIndexer` stores `CommitRecord` objects in Chroma for semantic queries like *"When did reranking get introduced?"*; incremental via `content_hash`, with per-repo pruning

### Memory System (Stage C)
- **Conversation memory** (C.1) — persistent short/medium-term memory per session: tracks `active_project`, rolling `current_topics`, and `recent_questions`; automatically injected into every RAG prompt and updated after each answer
- **Topic extraction** — LLM extracts 1-5 topic tags from each Q-A turn to keep `current_topics` up to date
- **Project inference** — periodically infers `active_project` from recent conversation history via LLM
- **SQLite-backed sessions** — all memory persists across restarts; multiple named sessions supported
- **Semantic user memory** (C.2) — long-term recency-weighted interest profile across all sessions; EMA scoring surfaces frequently and recently discussed topics; injected into every RAG prompt as a `Frequent Research Areas` hint
- **Knowledge timeline** (C.3) — daily activity log aggregating topics and retrieved document IDs per calendar day; enables queries like "what was I working on last week?"; injects a `Recent Activity` block into the prompt
- **Research session tracking** (C.4) — named research sessions that group related queries, retrieved documents, and free-form notes; an *active session* auto-accumulates every Q-A turn; session context injected into the RAG prompt; supports archive/restore lifecycle

### Architecture & Extensibility
- **Composition-first client API** — `LocalLlamaClient` is assembled from `RetrievalCapability`, `KnowledgeCapability`, `GenerationCapability`, and `IndexingCapability`; legacy methods remain as compatibility wrappers
- **Pluggable component providers** — `build_client_components(..., providers=...)` supports replacing model/storage/retrieval/knowledge/memory builders without editing central assembly code
- **Typed memory gateway contracts** — `MemoryGateway` provides typed commands (`StoreCommand`, `RetrieveQuery`, `ClearCommand`) while preserving legacy `scope + key` APIs for compatibility
- **UI protocol decoupling** — UI controllers now depend on narrow `Protocol` interfaces (`SearchClientProtocol`, `IndexClientProtocol`, `KnowledgeClientProtocol`) instead of directly depending on `LocalLlamaClient`

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
embed_base:   "http://localhost:8080/v1"   # API root URL (no trailing slash, no /embeddings suffix)
llm_base:     "http://localhost:8080/v1"
embed_model:  "text-embedding-ada-002"
llm_model:    "local-model"
api_key:      "sk-no-key-required"          # default fallback key
embed_api_key: ""                           # embedding API key (falls back to api_key)
llm_api_key:   ""                           # LLM API key (falls back to api_key)
llm_kwargs:  {}

# --- Provider ---
model_provider:      "openai"   # "openai" | "openrouter"
requests_rate_limit: 20         # max embedding requests/min (0 = unlimited; openrouter free tier ≈ 20)

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

The dashboard is an engineering tool for tuning retrieval — not a demo. It has six tabs:

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

**Knowledge tab**

- Run Stage B operations (enrich, QA generation, topic clustering, cross-doc linking) for any indexed `doc_id` directly from the UI
- Inspect per-chunk enrichment status and cluster assignments without running scripts

**Notion tab**

- Sync Notion pages to local RawStore (incremental or full), embed into Chroma, or run both in one click
- Full-sync checkbox ignores stored cursor and re-fetches all pages
- Status badges show pages seen / updated / skipped / errors after each operation
- Hybrid search and RAG query over synced Notion content, with configurable `k` / `fetch-k`
- Page list refreshes after every sync; shows page titles and sync status
- Shows "Notion not configured" when `notion_token` is not set

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

### Memory Gateway (Stage C.1 ~ C.4)

Memory is automatically injected during `answer_query()` through `RAGEngine`,
and can also be managed via the unified gateway.

```python
from rag.memory.gateway import ClearCommand, RetrieveQuery, StoreCommand

# C.1 conversation state
client.memory_manager.store_typed(
  StoreCommand(scope="conversation", key="active_project", value="Building Notion AI Knowledge System")
)
state = client.memory_manager.retrieve_typed(
  RetrieveQuery(scope="conversation", key="state")
)
print(state.active_project)
print(state.current_topics)

# C.2 user profile
client.memory_manager.store_typed(
  StoreCommand(scope="user_profile", key="topics", value=["RAG", "Memory"])
)
interests = client.memory_manager.retrieve_typed(
  RetrieveQuery(scope="user_profile", key="top_interests")
)

# C.3 timeline
client.memory_manager.store_typed(
  StoreCommand(scope="timeline", key="activity", value={"topics": ["RAG"], "doc_ids": ["doc-1"]})
)
recent = client.memory_manager.retrieve_typed(
  RetrieveQuery(scope="timeline", key="recent")
)

# C.4 research sessions
session = client.memory_manager.store_typed(
  StoreCommand(scope="research", key="session", value={"name": "Agentic RAG research", "tags": ["RAG"]})
)
client.memory_manager.store_typed(
  StoreCommand(scope="research", key="note", value={"content": "Investigate tool-augmented reasoning."})
)

# Legacy API is still supported for compatibility.
client.memory_manager.clear("conversation")
client.memory_manager.clear_typed(ClearCommand(scope="timeline"))
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
│   ├── knowledge_tab.py          # Knowledge tab UI — Stage B operations per doc_id
│   ├── notion_tab.py             # Notion tab UI — sync controls, page list, hybrid search / RAG
│   ├── config_tab.py             # Config tab UI — form-based schema-driven settings editor
│   ├── search_controller.py      # Search tab logic (state, search, filter)
│   ├── index_controller.py       # Index tab logic (save file, embed, list docs)
│   ├── knowledge_controller.py   # Knowledge tab logic (enrich, QA, cluster, link)
│   ├── client_protocols.py       # Narrow UI-facing client protocols (search/index/knowledge)
│   ├── notion_controller.py      # Notion tab logic (sync, embed, search, RAG query)
│   └── config_controller.py      # Config tab logic (load, save, hot-reload, schema)
├── etc/
│   └── config.yaml               # All runtime settings
├── rag/
│   ├── client.py                 # LocalLlamaClient — composition root + compatibility facade
│   ├── client_capabilities.py    # Retrieval/Knowledge/Generation/Indexing capability facades
│   ├── client_components.py      # Layered builders + pluggable provider assembly
│   ├── engine.py                 # RAGEngine — query expansion, retrieval, rerank, generation, memory
│   ├── indexer.py                # Indexer — SQLRecordManager lifecycle and document ingestion
│   ├── embeddings.py             # OpenRouterEmbeddings (multi-threaded httpx) + OpenAI wrapper
│   ├── rate_limiter.py           # RateLimiter — thread-safe sliding-window request throttle
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
│   │   ├── gateway.py            # Unified memory gateway + typed command/query contracts
│   │   ├── store.py              # MemoryStore — SQLite persistence for all memory subsystems (C.1–C.4)
│   │   ├── manager.py            # ConversationMemory — session lifecycle, topic extraction, prompt injection (C.1)
│   │   ├── user_memory.py        # UserMemoryManager — EMA-weighted long-term interest profile (C.2)
│   │   ├── timeline.py           # KnowledgeTimeline — daily activity log with topic + doc tracking (C.3)
│   │   └── research_session.py   # ResearchSessionManager — named sessions, queries, notes lifecycle (C.4)
│   ├── code/
│   │   ├── schema.py             # Domain models: RepoFile, RepoManifest, ManifestDiff, CommitInfo, FileSnapshot, SymbolEvolution, CommitRecord
│   │   ├── scanner.py            # RepoScanner — directory walk, manifest generation, incremental diff (GCR1.1)
│   │   ├── ast_parser.py         # PythonASTParser — libcst-based chunk + edge extraction (GCR1.2)
│   │   ├── symbol_store.py       # SymbolStore — in-memory symbol registry with JSON persistence (GCR1.3)
│   │   ├── indexer.py            # CodeIndexer — four-level Chroma indexing (repo/file/symbol/block) (GCR1.4)
│   │   ├── git_reader.py         # GitReader — git subprocess wrapper; CommitInfo + FileSnapshot builder (GCR1.5)
│   │   ├── snapshot_store.py     # SnapshotStore — temporal file history accumulator (GCR1.5)
│   │   ├── graph_store.py        # GraphStore — SQLite edge store for dependency + evolution data (GCR2.1/2.2)
│   │   ├── evolution_builder.py  # build_symbol_evolutions() — derives SymbolEvolution from snapshots (GCR2.2)
│   │   ├── diff_analyzer.py      # DiffAnalyzer — LLM-based one-line diff summariser (GCR2.3)
│   │   ├── commit_analyzer.py    # CommitAnalyzer — derives affected_symbols and LLM commit summary (GCR2.4)
│   │   └── commit_indexer.py     # CommitIndexer — Chroma storage and semantic search for commits (GCR2.4)
│   └── retrieval/
│       ├── bm25.py               # BM25Index + RRF fusion
│       └── filters.py            # SearchFilter — Chroma where-clause builder
├── utils/
│   ├── config.py                 # AppConfig — typed properties over config.yaml
│   ├── file_processor.py         # PDF chunking, YAML / JSON / CSV helpers
│   └── logger.py                 # AppLogger — centralised logging setup
├── testing/
│   ├── code/
│   │   ├── testing_base_chunk.py           # BaseChunk unit tests
│   │   ├── testing_ast_parser.py           # PythonASTParser unit tests (GCR1.2)
│   │   ├── testing_symbol_store.py         # SymbolStore unit tests (GCR1.3)
│   │   ├── testing_code_indexer.py         # CodeIndexer unit tests (GCR1.4)
│   │   ├── testing_collection_strategy.py  # collection strategy unit tests
│   │   ├── testing_git_reader.py           # GitReader unit tests (GCR1.5)
│   │   ├── testing_content_hash_diff.py    # incremental hash diff unit tests
│   │   ├── testing_scanner.py              # RepoScanner unit tests (GCR1.1)
│   │   ├── testing_dependency_graph.py     # GraphStore + edge extraction tests (GCR2.1)
│   │   ├── testing_symbol_evolution.py     # SymbolEvolution builder tests (GCR2.2)
│   │   ├── testing_diff_analysis.py        # DiffAnalyzer unit tests (GCR2.3)
│   │   ├── testing_commit_indexing.py      # CommitAnalyzer + CommitIndexer tests (GCR2.4)
│   │   └── testing_code_bm25.py            # code BM25 index tests
│   ├── ingest/
│   │   ├── testing_chunker.py              # chunker strategy tests (integration: Notion)
│   │   ├── testing_embedder.py             # NotionEmbedder tests (integration: Notion)
│   │   ├── testing_extractor.py            # KnowledgeExtractor unit tests (B.1)
│   │   ├── testing_qa.py                   # QAGenerator unit tests (B.2)
│   │   ├── testing_clusterer.py            # TopicClusterer unit tests (B.3)
│   │   └── testing_linker.py               # CrossDocLinker unit tests (B.4)
│   ├── knowledge/
│   │   ├── testing_knowledge.py            # RawStore + NotionClient conversion tests
│   │   └── testing_models.py               # domain model unit tests
│   ├── notion/
│   │   ├── testing_notion_api.py           # Notion REST API integration tests
│   │   ├── testing_blocks.py               # block-level parsing tests
│   │   ├── testing_sync.py                 # NotionSyncPipeline integration tests
│   │   └── testing_pipeline.py             # NotionRAGClient end-to-end tests
│   ├── memory/
│   │   ├── testing_memory.py               # ConversationMemory unit tests (C.1)
│   │   ├── testing_user_memory.py          # UserMemoryManager + timeline tests (C.2/C.3)
│   │   └── testing_research_session.py     # ResearchSessionManager unit tests (C.4)
│   ├── retrieval/
│   │   ├── testing_base_indexer.py         # base indexer unit tests
│   │   ├── testing_unified_engine.py       # unified retrieval engine tests
│   │   └── testing_retrieval_pipeline.py   # end-to-end retrieval pipeline tests
│   ├── eval/
│   │   ├── testing_eval.py                 # retrieval evaluation helpers (integration)
│   │   └── testing_eval_unit.py            # evaluation metric unit tests
│   └── config/
│       ├── testing_llm_config.py           # LLM / provider config smoke tests
│       └── testing_openrouter.py           # OpenRouterEmbeddings rate-limit tests
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

