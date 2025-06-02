# PaperQA Search API: Key Functions and Logic

## Key Classes and Functions

### `SearchIndex`
- **Purpose:**  
  A wrapper around a Tantivy-based index, providing high-level async methods for indexing, storing, and searching documents and their chunks.
- **Key Methods:**
  - `__init__`: Initialize with fields, index name, directory, and storage.
  - `add_document`: Add a document (with metadata and content) to the index, with deduplication and file lock retry logic.
  - `searcher`: Async property to get a Tantivy searcher for querying.
  - `writer`: Async context manager for writing to the index.
  - `index_files`: Loads a mapping of file locations to file hashes for deduplication.
  - `filecheck`: Checks if a file (by hash) is already indexed.
  - `mark_failed_document`: Mark a document as failed in the index.
  - `count`: Returns the number of documents in the index.

### `get_directory_index`
- **Purpose:**  
  Build or load a `SearchIndex` from a directory of text files (PDF, TXT, HTML, MD).
- **Key Logic:**
  - Reads the directory, finds valid files, and builds the index.
  - Optionally synchronizes the index with the directory (adding/removing files as needed).
  - Uses a manifest file for additional metadata if present.
  - Supports concurrent processing of files for efficient indexing.

## High-Level Logic Flow

```mermaid
flowchart TD
    A[Initialize SearchIndex] --> B[Build/Load Tantivy Schema]
    B --> C[Create/Load Index Directory]
    C --> D[Add Document]
    D --> E[Check for Duplicates]
    E -- Not Duplicate --> F[Write Document Metadata to Index]
    F --> G[Store Document Content (optional)]
    G --> H[Update File Hash Mapping]
    D --> I[Search]
    I --> J[Load Searcher]
    J --> K[Query Index for Relevant Chunks]
    K --> L[Return Results]
```

## Usage Example

```python
index = SearchIndex(fields=["file_location", "body", "title"])
await index.add_document({"file_location": "paper1.pdf", "body": "text...", "title": "A Study"})
results = await index.searcher.search(query)
```

### Deleting a Document
```python
await index.delete_document("paper1.pdf")
```

### Querying the Index
```python
results = await index.query(
    query="machine learning",
    top_n=5,
    min_score=0.2,
    field_subset=["body", "title"]
)
for doc in results:
    print(doc)
```

### Saving the Index
```python
await index.save_index()
```

### Removing a Document from the Index
```python
await index.remove_from_index("paper1.pdf")
```

### Loading a SearchIndex from a Directory
```python
from paperqa.agents.search import get_directory_index
search_index = await get_directory_index(settings=my_settings)
```
