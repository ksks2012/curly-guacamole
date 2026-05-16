"""
Prompt templates for the RAG pipeline.

Centralising all prompts here makes it easy to:
  - A/B test different phrasings without touching business logic
  - Add new pipeline stages (query expansion, hypothetical doc embedding, etc.)
  - Switch language / persona by swapping the module-level constants

Available templates
-------------------
RAG_PROMPT                  : final answer generation (context + question → answer)
MEMORY_AWARE_RAG_PROMPT     : RAG_PROMPT with {memory_context} injected above context
QUERY_EXPANSION_PROMPT      : generates N alternative phrasings of a query
KNOWLEDGE_EXTRACTION_PROMPT : extracts structured knowledge from a text chunk
QA_GENERATION_PROMPT        : generates N Q-A pairs from a text chunk
TOPIC_LABEL_PROMPT          : names a cluster of document excerpts
TOPIC_EXTRACTION_PROMPT     : extracts 1-5 topic tags from a Q-A turn (C.1)
PROJECT_INFERENCE_PROMPT    : infers the active project from recent conversation (C.1)
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

# ---------------------------------------------------------------------------
# Knowledge extraction prompt
#
# Variables: {text}
# Used to extract structured knowledge artifacts from a single text chunk.
# The LLM must return a single valid JSON object so the caller can parse it.
#
# Prefix ka_ (knowledge artifact) is documented here; the prompt itself does
# not use the prefix — that is applied by KnowledgeExtractor.
# ---------------------------------------------------------------------------
KNOWLEDGE_EXTRACTION_PROMPT = PromptTemplate.from_template("""\
Extract structured knowledge from the text chunk below.

Reply ONLY with a single valid JSON object using exactly these keys:
{{
  "summary":   "2-3 sentence summary of what this chunk is about (30-80 words)",
  "keywords":  ["key term or phrase", "..."],
  "entities":  ["named person / place / org / product / technology", "..."],
  "topics":    ["broad topic or theme", "..."],
  "questions": ["natural question whose answer is in this chunk", "..."]
}}

Guidelines:
- summary  : concise and informative; do not start with "This chunk…"
- keywords : 3-8 specific terms critical for retrieval; prefer noun phrases
- entities : proper nouns only; skip generic words
- topics   : 1-4 high-level categories (e.g. "machine learning", "pricing")
- questions: 2-5 questions a user might ask whose answer appears in this text

Text:
{text}
""")

# ---------------------------------------------------------------------------
# QA pair generation prompt
#
# Variables: {text}, {n}
# Used by QAGenerator to produce N question-answer pairs from a single chunk.
# Questions are stored as page_content in the QA index so that user queries
# hit generated questions semantically rather than raw chunk text.
# ---------------------------------------------------------------------------
QA_GENERATION_PROMPT = PromptTemplate.from_template("""\
You are a question-answer generation assistant.

Given the text chunk below, generate {n} question-answer pairs.
Each question should be something a user might naturally ask whose answer is
clearly found in the text.  Each answer must be concise (1-3 sentences) and
derived directly from the text — do not add outside knowledge.

Reply ONLY with a JSON array using exactly this format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Do NOT include any explanation or text outside the JSON array.

Text:
{text}
""")

# ---------------------------------------------------------------------------
# Topic label prompt
#
# Variables: {text}
# Used by TopicClusterer._label_cluster() to name a cluster of documents.
# The LLM receives the most representative chunks (closest to the centroid)
# and must return a single short snake_case identifier, e.g. "topic_rag".
# ---------------------------------------------------------------------------
TOPIC_LABEL_PROMPT = PromptTemplate.from_template("""\
You are a topic labeling assistant.

Read the document excerpts below and identify the single dominant topic they share.

Reply with ONLY a short snake_case label starting with "topic_", for example:
  topic_rag
  topic_evolutionary_algorithm
  topic_agents
  topic_llm_fine_tuning

Rules:
- One label only — no explanation, no JSON, no extra text.
- Use lowercase letters, digits, and underscores only.
- Keep it under 40 characters.

Excerpts:
{text}
""")

# ---------------------------------------------------------------------------
# Conversation Memory prompts (Stage C.1)
# ---------------------------------------------------------------------------

# TOPIC_EXTRACTION_PROMPT
# Variables: {question}, {answer_excerpt}
# Called by ConversationMemory.add_turn() after each Q-A exchange.
# Returns a JSON array of 1-5 short topic strings.
# ---------------------------------------------------------------------------
TOPIC_EXTRACTION_PROMPT = PromptTemplate.from_template("""\
You are a topic tagging assistant.

Given the question and answer below, extract 1-5 concise topic tags that best
describe what was discussed.

Reply ONLY with a JSON array of short topic strings (2-5 words each), e.g.:
["RAG Architecture", "Vector Search", "Knowledge Extraction"]

Rules:
- Return ONLY the JSON array — no explanation, no extra text.
- Each tag should be a noun phrase, 2-5 words, title-cased.
- If no clear topic can be identified, return: []

Question: {question}

Answer (excerpt): {answer_excerpt}
""")

# ---------------------------------------------------------------------------
# PROJECT_INFERENCE_PROMPT
# Variables: {questions}, {topics}
# Called periodically by ConversationMemory._refresh_active_project().
# Returns a single plain-text project description (no JSON).
# ---------------------------------------------------------------------------
PROJECT_INFERENCE_PROMPT = PromptTemplate.from_template("""\
You are a project context analyst.

Based on the following recent questions and discussion topics from a conversation,
infer the user's primary active project or goal in 6-15 words.

Rules:
- Return ONLY the project description as a plain string.
- No quotes, no JSON, no explanation.
- Be specific and descriptive, e.g. "Building Notion AI knowledge management system with RAG"

Recent topics: {topics}

Recent questions:
{questions}
""")

# ---------------------------------------------------------------------------
# MEMORY_AWARE_RAG_PROMPT
# Variables: {memory_context}, {context}, {question}
# Used by RAGEngine when ConversationMemory is active.
# {memory_context} is produced by ConversationMemory.build_context_block().
# ---------------------------------------------------------------------------
MEMORY_AWARE_RAG_PROMPT = PromptTemplate.from_template("""\
You are a document analysis assistant with conversation memory.

=== Conversation Memory ===
{memory_context}
===========================

Rules:
- Answer ONLY using the provided document context below.
- If the answer is not in the context, say "I don't know based on the provided documents."
- You MUST cite the source for every claim, e.g. [page 3, test.pdf].
- Use the conversation memory to personalise and contextualise your answer when relevant.

Context:
{context}

Question:
{question}
""")
