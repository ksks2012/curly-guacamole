import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_processor import read_yaml

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

    def similarity_search(self, query: str, k: int = 4):
        """Returns a list of similar documents from Chroma (LangChain Document objects)."""
        return self.db.similarity_search(query, k=k)

    def answer_query(self, query: str, k: int = 1):
        """Uses retrieved documents as context to generate a response (simple prompt concatenation)."""
        docs = self.similarity_search(query, k=k)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = f"Answer based on the following information:\n{context}\n\nQuestion: {query}"
        # Use invoke or __call__ depending on your ChatOpenAI implementation
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(prompt)
        return self.llm(prompt)

    def add_texts(self, texts, metadatas=None, ids=None):
        """Adds texts to Chroma (uses available Chroma API method)."""
        if hasattr(self.db, "add_texts"):
            return self.db.add_texts(texts, metadatas=metadatas, ids=ids)
        # fallback: use from_texts with persist (heavier operation)
        self.db = Chroma.from_texts(texts, embedding=self.embed, persist_directory=self.persist_directory)

    def add_pdf(self, pdf_path):
        """Loads a PDF and adds its content to Chroma."""
        # Demonstrates the structure; use PyPDF2, pdfplumber, etc. for actual PDF parsing
        try:
            loader = PyPDFLoader(pdf_path)
            data = loader.load()
            print(f"Loaded {len(data)} pages.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = text_splitter.split_documents(data)
            print(f"Created {len(chunks)} chunks.")

            self.db = Chroma.from_documents(chunks, embedding=self.embed, persist_directory=self.persist_directory)
        except Exception as e:
            print(f"Error loading PDF: {e}")


    def persist(self):
        """Force write to disk (newer Chroma versions auto-persist when persist_directory is set)."""
        pass  # langchain_chroma >= 0.1 auto-persists when persist_directory is set

def main():
    client = LocalLlamaClient()
    client.add_pdf(pdf_path)
    docs = client.similarity_search(_config.get("test_search", "test"))
    resp = client.answer_query(_config.get("test_query", "What are the main contents of this document?"))
    print(resp)

if __name__ == "__main__":
    main()