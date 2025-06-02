# How Document Search Works in PaperQA

## Overview
PaperQA combines traditional keyword search (using Tantivy) and modern embedding-based (vector) search to provide high-accuracy retrieval from scientific documents. It also leverages rich metadata (such as citation count) to enhance search, ranking, and answer generation.

---

## 1. Indexing: Preparing for Search

When documents (PDFs, text files) are added to PaperQA:
- **Text is extracted and chunked** into manageable pieces.
- **Metadata** (title, authors, year, citation count, etc.) is extracted using external providers (e.g., Crossref, Semantic Scholar) and stored alongside each chunk.
- **Keyword Indexing:**
  - Each chunk is indexed using Tantivy, a fast full-text search engine, allowing for efficient keyword-based retrieval.
- **Embedding Indexing:**
  - Each chunk is also embedded (converted to a vector) using a configurable embedding model.
  - These vectors are stored in a vector store (e.g., in-memory, Qdrant, etc.) for semantic similarity search.

---

## 2. Search: Combining Keyword and Embedding Search

When a user asks a question, PaperQA performs a **hybrid retrieval** process:

### Step 1: Keyword Search (Tantivy)
- The question is parsed and used to query the Tantivy index.
- This retrieves chunks that match the query terms (e.g., papers mentioning "antibodies").
- This step is fast and can handle Boolean logic, field filters, etc.

### Step 2: Embedding Search (Vector Search)
- The question is embedded into a vector.
- The vector store is queried for chunks whose embeddings are most similar to the question embedding (semantic search).
- This step finds relevant content even if the exact keywords are not present.

### Step 3: Maximal Marginal Relevance (MMR) and Hybrid Ranking
- PaperQA can use **Maximal Marginal Relevance (MMR)** to combine the results of keyword and embedding search.
- MMR balances relevance (similarity to the query) and diversity (reducing redundancy among results).
- The top-k chunks are selected for further processing.

```mermaid
flowchart TD
    Q[User Query] --> KWS[Keyword Search (Tantivy)]
    Q --> EBS[Embedding Search (Vector)]
    KWS & EBS --> MMR[MMR/Hybrid Ranking]
    MMR --> C[Top-k Chunks]
```

---

## 3. Using Metadata (e.g., Citation Count)

- **Metadata Extraction:**
  - When documents are added, PaperQA uses providers like Crossref and Semantic Scholar to fetch metadata, including citation count, publication year, journal, etc.
  - This metadata is stored in `DocDetails` objects and associated with each chunk.
- **How Metadata is Used:**
  - **Ranking:** Chunks or documents with higher citation counts or from higher-quality journals can be ranked higher or preferred in evidence selection.
  - **Filtering:** Users or agents can filter results by year, author, or citation count.
  - **Citations in Answers:** When generating answers, PaperQA includes citations and can mention citation counts (e.g., "This article has 120 citations").
  - **Contextual Prompts:** Metadata can be included in prompts to the LLM to improve answer quality and trustworthiness.

---

## 4. Answer Generation
- The top-k evidence chunks (with their metadata) are summarized and re-ranked by an LLM.
- The LLM generates a final answer, citing the most relevant sources and including metadata (like citation count) as needed.

---

## 5. Summary Table
| Component         | Role in Search Pipeline                                      |
|-------------------|-------------------------------------------------------------|
| Tantivy           | Fast keyword search, initial candidate retrieval             |
| Embedding Search  | Semantic retrieval, finds relevant content beyond keywords   |
| MMR/Hybrid        | Combines keyword and embedding results for diversity/relevance|
| Metadata          | Used for ranking, filtering, and citation in answers         |

---

## References
- [PaperQA Documentation](https://github.com/Future-House/paper-qa)
- Code: see `paperqa_code.md` for implementation details 