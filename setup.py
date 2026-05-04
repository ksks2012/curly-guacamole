from setuptools import setup, find_packages

setup(
    name="curly-guacamole",
    version="0.1.0",
    description="Local LLM RAG client using LangChain, Chroma, and OpenAI-compatible servers",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langchain-openai",
        "langchain-chroma",
        "langchain-community",
        "langchain-text-splitters",
        "langchain",
        "chromadb",
        "pypdf",
        "pyyaml",
    ],
    include_package_data=True,
    package_data={
        "": ["etc/*.yaml"],
    },
)
