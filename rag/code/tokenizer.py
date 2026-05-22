"""
rag/code/tokenizer.py — Symbol-aware tokeniser for code BM25 indexing.

Standard prose tokenisers treat "RAGEngine" as one opaque token, so a query
for "engine" would miss it.  Code identifiers carry semantics in their
sub-words; this tokeniser extracts them explicitly.

Rules (applied in order)
--------------------------
1. Split on any non-alphanumeric/non-underscore character
   (dots, brackets, angle-brackets, colons, slashes, …)
2. Strip leading and trailing underscores from each part
   (``_DEFAULT`` → ``DEFAULT``, ``__init__`` → ``init``)
3. Split snake_case parts on ``_``
4. Split each fragment on camelCase and consecutive-uppercase boundaries
   using regex (``RAGEngine`` → ``RAG``, ``Engine``;
               ``BM25Index`` → ``BM``, ``25``, ``Index``)
5. Lowercase all tokens and filter empty strings

Examples
--------
>>> code_tokenize("RAGEngine.retrieve")
['rag', 'engine', 'retrieve']
>>> code_tokenize("content_hash")
['content', 'hash']
>>> code_tokenize("_DEFAULT_COLLECTION_NAMES")
['default', 'collection', 'names']
>>> code_tokenize("BM25Index")
['bm', '25', 'index']
>>> code_tokenize("index()")
['index']
>>> code_tokenize("SearchFilter::to_chroma")
['search', 'filter', 'to', 'chroma']
"""

from __future__ import annotations

import re

# Separator: everything that is not alphanumeric or underscore
_SEP_RE = re.compile(r"[^a-zA-Z0-9_]+")

# camelCase / consecutive-caps / number splitter.
# Matches (in order):
#   ABCDef  → ABC  Def   (consecutive caps before a lowercase-start word)
#   camelId → camel  Id  (standard camelCase)
#   ALL     → ALL        (all-caps word)
#   123     → 123        (numeric run)
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def code_tokenize(text: str) -> list[str]:
    """Tokenise a code identifier or mixed prose/code string into sub-word tokens.

    Suitable for BM25 indexing of code symbol names, file paths, docstrings,
    and any text that mixes natural language with code identifiers.

    Parameters
    ----------
    text : Input string (symbol name, function signature, docstring, …).

    Returns
    -------
    List of lowercase sub-word tokens with no duplicates in order.
    """
    tokens: list[str] = []

    for raw_part in _SEP_RE.split(text):
        # Strip leading/trailing underscores (handles dunder names, _CONST, etc.)
        part = raw_part.strip("_")
        if not part:
            continue

        # snake_case split
        for snake_part in part.split("_"):
            frag = snake_part.strip("_")
            if not frag:
                continue

            # camelCase / consecutive-caps / number split
            camel_tokens = _CAMEL_RE.findall(frag)
            if camel_tokens:
                tokens.extend(t.lower() for t in camel_tokens)
            else:
                # fallback — plain lowercase (e.g. already all-lower)
                tokens.append(frag.lower())

    # Filter empty strings (shouldn't happen but be safe)
    return [t for t in tokens if t]
