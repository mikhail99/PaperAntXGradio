# `paperqa.docs.py` — Important Methods Overview

This document provides concise explanations of the most important methods in `paperqa/docs.py`. For methods with both synchronous and asynchronous versions, only the async version is described.

---

## Class: `Docs`
A collection of documents and their chunked texts, supporting addition, indexing, search, and retrieval for question answering.

### Document Management
- **`async def aadd_file(...)`**
  - Adds a document from a file-like object (PDF, text, HTML) to the collection. Handles temporary file creation and delegates to `aadd`.

- **`async def aadd_url(...)`**
  - Downloads a document from a URL and adds it to the collection using `aadd_file`.

- **`async def aadd(...)`**
  - Adds a document from a file path. Handles citation extraction, metadata parsing, chunking, and embedding. Adds the document and its chunks to the collection.

- **`async def aadd_texts(...)`**
  - Adds pre-chunked texts (with embeddings) to the collection, associating them with a document.

- **`def delete(...)`**
  - Removes a document and its chunks from the collection by name, docname, or dockey.

- **`def clear_docs()`**
  - Removes all documents and chunks from the collection.

### Indexing and Embedding
- **`async def _build_texts_index(embedding_model)`**
  - Ensures all texts are embedded and added to the vector index. Embeds any missing chunks as needed.

### Search & Retrieval
- **`async def retrieve_texts(query, k, ...)`**
  - Performs hybrid search (keyword + embedding) using Maximal Marginal Relevance (MMR) to retrieve the top-k most relevant and diverse chunks for a query.

- **`async def aget_evidence(query, ...)`**
  - Main entry point for evidence retrieval. Given a user query, retrieves top-k relevant chunks (using `retrieve_texts`), summarizes them, and attaches them to the session context for downstream answer generation.

- **`async def aquery(query, ...)`**
  - Main entry point for full question answering. If no context is present, calls `aget_evidence` to retrieve evidence, then generates an answer using the LLM and the retrieved context.

### Utility & Internal
- **`def _get_unique_name(docname)`**
  - Ensures document names are unique within the collection.

- **`def handle_default(cls, value, info)`**
  - Pydantic validator for setting the default index path.

---

## Typical Usage Flow
1. **Add documents** using `aadd_file`, `aadd_url`, or `aadd`.
2. **Ask a question** using `aquery` (for full QA) or `aget_evidence` (for evidence retrieval only).
3. **Retrieve results** from the returned session object (contexts, answer, references, etc.).

---

For more details, see the full code or the [PaperQA documentation](https://github.com/Future-House/paper-qa). 