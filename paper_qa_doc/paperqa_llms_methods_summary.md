# `paperqa.llms.py` — Important Methods & Classes Overview

This document provides concise explanations of the most important classes and methods in `paperqa/llms.py`. For methods with both synchronous and asynchronous versions, only the async version is described.

---

## Core Classes

### `VectorStore` (Abstract Base Class)
Defines the interface for vector storage and retrieval, supporting similarity and MMR search.

- **`async def add_texts_and_embeddings(texts)`**
  - Adds texts and their embeddings to the store. Updates internal hashes.

- **`async def similarity_search(query, k, embedding_model)`**
  - Returns the top-k most similar texts to the query using the provided embedding model.

- **`async def partitioned_similarity_search(query, k, embedding_model, partitioning_fn)`**
  - (Optional) Performs similarity search within partitions of the data, as defined by `partitioning_fn`.

- **`async def max_marginal_relevance_search(query, k, fetch_k, embedding_model, partitioning_fn=None)`**
  - Performs Maximal Marginal Relevance (MMR) search, combining relevance and diversity to select top-k results. If `partitioning_fn` is provided, applies MMR within each partition.

- **`def clear()`**
  - Clears all stored texts and hashes from the vector store.

---

### `NumpyVectorStore` (Concrete Implementation)
In-memory vector store using NumPy arrays for fast similarity search.

- **`async def add_texts_and_embeddings(texts)`**
  - Adds texts and their embeddings to the store and updates the internal NumPy matrix.

- **`async def similarity_search(query, k, embedding_model)`**
  - Computes cosine similarity between the query and all stored embeddings, returning the top-k most similar texts and their scores.

- **`async def partitioned_similarity_search(query, k, embedding_model, partitioning_fn)`**
  - Splits texts into partitions, performs similarity search in each, and interleaves results for diversity.

- **`def clear()`**
  - Removes all texts and embeddings from the store.

---

### `QdrantVectorStore` (Concrete Implementation)
Vector store backed by a Qdrant database (local or remote), supporting persistent and scalable storage.

- **`async def add_texts_and_embeddings(texts)`**
  - Adds texts and their embeddings to a Qdrant collection, creating the collection if needed.

- **`async def similarity_search(query, k, embedding_model)`**
  - Queries Qdrant for the top-k most similar vectors to the query embedding.

- **`async def aclear()`**
  - Asynchronously deletes the Qdrant collection and clears internal state.

- **`@classmethod async def load_docs(...)`**
  - Loads all documents and their vectors from a Qdrant collection into a new `Docs` object.

- **`def clear()`**
  - Synchronous wrapper for `aclear`, runs in a separate thread to avoid event loop issues.

---

### Utility Functions

- **`def cosine_similarity(a, b)`**
  - Computes the cosine similarity between two arrays of vectors.

- **`def embedding_model_factory(embedding: str, **kwargs)`**
  - Factory function to create an appropriate embedding model based on a string identifier. Supports hybrid, SentenceTransformer, LiteLLM, and sparse models.

---

## Typical Usage Flow
1. **Create a vector store** (e.g., `NumpyVectorStore` or `QdrantVectorStore`).
2. **Add texts and embeddings** using `add_texts_and_embeddings`.
3. **Perform search** using `similarity_search` or `max_marginal_relevance_search` for hybrid/MRR retrieval.

---

For more details, see the full code or the [PaperQA documentation](https://github.com/Future-House/paper-qa). 