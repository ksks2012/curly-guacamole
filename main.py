import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from utils.file_processor import read_yaml, load_and_chunk_pdf

# ---------------------------------------------------------------------------
# RAG prompt: forces the model to cite sources and stay grounded in context.
# {context} is pre-formatted with [page N / filename] tags per chunk.
# {question} is the user query.
# ---------------------------------------------------------------------------
RAG_PROMPT = PromptTemplate.from_template("""\
You are a document analysis assistant.

Rules:
- Answer ONLY using the provided context below.
- If the answer is not in the context, say "I don't know based on the provided documents."
- You MUST cite the source for every claim, e.g. [page 3, test.pdf].

Context:
{context}

Question:
{question}
""")

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etc", "config.yaml")
_config = read_yaml(_CONFIG_PATH)

pdf_path = _config.get("pdf_path", "./data/test.pdf")

class LocalLlamaClient:
    """
    Wraps a local embedding server (OpenAI-compatible), a Chroma vector store, and a local LLM (OpenAI-compatible).
    """
    def __init__(
        self,
        embed_base: str = _config.get("embed_base", "http://localhost:8080/v1/"),
        llm_base: str = _config.get("llm_base", "http://localhost:8080/v1/"),
        embed_model: str = _config.get("embed_model", "text-embedding-ada-002"),
        llm_model: str = _config.get("llm_model", "local-model"),
        persist_directory: str = _config.get("persist_directory", "./my_db"),
        api_key: str = _config.get("api_key", "sk-no-key-required"),
        **llm_kwargs
    ):
        # Embedding: points to your embedding server (llama.cpp server)
        self.embed = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=embed_base,
            model=embed_model
        )

        # Vector store (Chroma)
        self.persist_directory = persist_directory
        self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embed)

        # LLM (chat) - points to your LLM server (also OpenAI-compatible)
        self.llm = ChatOpenAI(
            base_url=llm_base,
            api_key=api_key,
            model=llm_model,
            **llm_kwargs
        )

    def get_retriever(self, k: int = 5, fetch_k: int = 20):
        """
        Returns an MMR retriever.
          k       : number of documents returned to the LLM
          fetch_k : candidate pool size before MMR re-ranking (larger = more diverse / slower)
        """
        return self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k},
        )

    def similarity_search(self, query: str, k: int = 4):
        """Returns a list of similar documents from Chroma (LangChain Document objects)."""
        return self.db.similarity_search(query, k=k)

    def answer_query(self, query: str, k: int = 5, fetch_k: int = 20):
        """Uses MMR retrieval to fetch documents, then generates a citation-grounded response."""
        retriever = self.get_retriever(k=k, fetch_k=fetch_k)
        docs = retriever.invoke(query)

        # Build context blocks with source tags so the LLM can cite them
        context_blocks = []
        for doc in docs:
            page = doc.metadata.get("page", "?")  # 0-based page from PyPDFLoader
            filename = doc.metadata.get("filename", "unknown")
            tag = f"[page {page + 1}, {filename}]"  # convert to 1-based for display
            context_blocks.append(f"{tag}\n{doc.page_content}")
        context = "\n\n".join(context_blocks)

        prompt = RAG_PROMPT.format(context=context, question=query)
        return self.llm.invoke(prompt)

    def add_texts(self, texts, metadatas=None, ids=None):
        """Adds texts to Chroma (uses available Chroma API method)."""
        if hasattr(self.db, "add_texts"):
            return self.db.add_texts(texts, metadatas=metadatas, ids=ids)
        # fallback: use from_texts with persist (heavier operation)
        self.db = Chroma.from_texts(texts, embedding=self.embed, persist_directory=self.persist_directory)

    def add_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """Loads a PDF, splits it into chunks with metadata, and indexes them in Chroma."""
        try:
            chunks = load_and_chunk_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            print(f"Loaded PDF: {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
            print(f"Sample metadata: {chunks[0].metadata if chunks else 'n/a'}")

            self.db = Chroma.from_documents(chunks, embedding=self.embed, persist_directory=self.persist_directory)
        except Exception as e:
            print(f"Error loading PDF: {e}")


    def persist(self):
        """Force write to disk (newer Chroma versions auto-persist when persist_directory is set)."""
        pass  # langchain_chroma >= 0.1 auto-persists when persist_directory is set

def main():
    client = LocalLlamaClient()
    # client.add_pdf(pdf_path)
    docs = client.similarity_search(_config.get("test_search", "test"))
    resp = client.answer_query(
        _config.get("test_query", "What are the main contents of this document?"),
        k=5,
        fetch_k=20,
    )
    print(resp)

if __name__ == "__main__":
    main()