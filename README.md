# langchain-test

A local RAG (Retrieval-Augmented Generation) client built with LangChain, Chroma, and any OpenAI-compatible server (e.g. llama.cpp).

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

Edit `etc/config.yaml` to set your server endpoints and model names:

```yaml
embed_base: "http://localhost:8080/v1/"
llm_base: "http://localhost:8080/v1/"
embed_model: "text-embedding-ada-002"
llm_model: "local-model"
persist_directory: "./my_db"
api_key: "sk-no-key-required"
pdf_path: "/path/to/your/document.pdf"
test_search: "your search keyword"
test_query: "What are the main contents of this document?"
```

## Usage

```bash
python main.py
```

`main.py` will:
1. Load a PDF and index it into the Chroma vector store.
2. Run a similarity search using `test_search`.
3. Answer the question defined in `test_query` using the retrieved context.

## Project Structure

```
.
├── etc/
│   └── config.yaml       # Configuration file
├── utils/
│   ├── __init__.py
│   └── file_processor.py # YAML / JSON / CSV read-write helpers
├── my_db/                # Chroma vector store (auto-created)
├── main.py               # Entry point
├── setup.py
└── requirements.txt
```
