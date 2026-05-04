from utils.config import AppConfig
from rag.client import LocalLlamaClient


def main():
    config = AppConfig()
    client = LocalLlamaClient(config)
    client.add_pdf(config.pdf_path)
    resp = client.answer_query(config.test_query, k=5, fetch_k=20)
    print(resp)


if __name__ == "__main__":
    main()
