"""
Prompt templates for the RAG pipeline.

Centralising all prompts here makes it easy to:
  - A/B test different phrasings without touching business logic
  - Add new pipeline stages (query expansion, hypothetical doc embedding, etc.)
  - Switch language / persona by swapping the module-level constants

Available templates
-------------------
RAG_PROMPT              : final answer generation (context + question → answer with citations)
QUERY_EXPANSION_PROMPT  : generates N alternative phrasings of a query to broaden retrieval
"""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# RAG answer prompt
#
# Variables: {context}, {question}
# {context} must be pre-formatted with [page N, filename] citation tags.
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

# ---------------------------------------------------------------------------
# Query expansion prompt
#
# Variables: {question}, {n}
# Used to generate `n` semantically varied re-phrasings of the original query.
# Each phrasing is sent to the vector store independently; results are merged
# and de-duplicated before reranking, improving recall over single-query retrieval.
#
# The LLM must return a plain JSON array so the caller can parse it reliably.
# ---------------------------------------------------------------------------
QUERY_EXPANSION_PROMPT = PromptTemplate.from_template("""\
You are a query expansion assistant.

Generate {n} alternative phrasings of the question below to improve document \
retrieval coverage. Each version should use different wording or a different \
perspective while preserving the original intent.

Original question: {question}

Reply ONLY with a JSON array of strings, one string per phrasing, e.g.:
["...", "...", "..."]

Do NOT include the original question. Do not add any other text.
""")
