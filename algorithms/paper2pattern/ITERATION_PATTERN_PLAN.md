# Implementation Plan: Iterative Pattern Extraction

This document outlines the development steps for the paper pattern extraction project. The focus is on small, iterative cycles with clear validation, keeping the codebase minimal and experimental.

### Iteration 1: Project Setup and Basic Paper Processing

**Goal:** Establish the project structure and reliably convert PDF papers into analyzable text sections.

- [ ] **Setup Project Structure:** Create the initial directory layout. A per-paper folder structure is best to keep related artifacts together.
  - [x] `/papers/[id]/paper.pdf`: To store input PDF.
  -  [x] `/papers/[id]/paper.html`: To store input HTML.
  - `/papers/[id]/metadata.json`: For paper metadata (title, abstract, authors).
  - `/outputs/[id]/full_text.txt`: The raw extracted text.
  - `/outputs/[id]/sections.json`: The paper's text split into sections.
  - `/outputs/[id]/dictionary.json`: The extracted domain dictionary.
  - `/outputs/[id]/pattern.json`: The final extracted pattern.
  - [x] `/src`: For all Python source code.
  - [x] `/db`: For the persistent ChromaDB vector store.
  - [x] `main.py`: A main script to run the processing pipelines on a single paper.

- [ ] **Curate Paper Collection:** Select 3-5 related arXiv papers. For each paper, create a directory in `/papers` named after its arXiv ID (e.g., `/papers/arXiv:2502.18857/`).
    - [x] Place the downloaded PDF as `paper.pdf` inside this directory.
    - [ ] Create a `metadata.json` file with the paper's title and abstract.

- [x] **Implement PDF to Text:** Create a Python function in `/src/pdf2txt_preprocessing.py` that takes all PDFs from `/papers` (format is `/papers/[id]/paper.pdf`) and if text is missing converts it to clean text (PyMuPDF), and saves it to `/outputs/[id]/full_text.txt`.

- [x] **Implement Section Splitting for html:** In `preprocessing.py`, create a function that takes a paper ID, reads paper in html format , splits it into sections, and saves the result to `/outputs/[arxiv_id]/sections.json`.
- [ ] **Implement Section Splitting for txt:** In `preprocessing.py`, create a function that takes a paper ID, reads paper in txt format `full_text.txt`, splits it into sections, and saves the result to `/outputs/[arxiv_id]/sections.json`.


- [ ] **Validation:**
    - Run the preprocessing functions for one paper ID.
    - Manually inspect the output files in `/outputs/[paper_id]/` to confirm the text is clean and the sections are split logically.

### Iteration 2: First DSPy Module - Domain Dictionary

**Goal:** Use `dspy` to perform the first, most basic LLM extraction task: identifying key terms.

- [x] **Setup DSPy:**
    - Install necessary packages: `pip install dspy openai chromadb`.
    - Configure `dspy` with an LLM (e.g., `dspy.OpenAI` or a local model) and set the API key.

- [x] **Define DSPy Signature:** In `/src/signatures.py`, define a `dspy.Signature` for extracting a list of domain terms and their definitions from a text.
    - `DomainDictionarySignature(paper_section -> domain_dictionary)` where `domain_dictionary` is a string containing a list of "Term: Definition".

- [x] **Create DSPy Module:** In `/src/modules.py`, create a `DictionaryExtractor(dspy.Module)` that uses `dspy.ChainOfThought` with the `DomainDictionarySignature`.

- [x] **Update Main Script:** Modify `main.py` to:
    1. Take a paper ID as an argument.
    2. Load the paper's "Methodology" section from `/outputs/[paper_id]/sections.json`.
    3. Pass the section to the `DictionaryExtractor`.

- [ ] **Validation:**
    - Run the script on one paper.
    - Manually review the extracted dictionary. Are the terms relevant? Are the definitions plausible? Perfection is not required, but the output should be reasonable.

### Iteration 3: Full Standalone Pattern Extraction

**Goal:** Expand the `dspy` program to extract all fields of a design pattern from a single paper, without external knowledge.

- [ ] **Define Full Pattern Signature:** In `signatures.py`, create a `PatternSignature` that defines all fields we want to extract (Intent, Motivation, Applicability, etc.). The `Structure` field should explicitly request a Mermaid.js diagram string as its output.

- [ ] **Create Full Extractor Module:** In `modules.py`, create a `StandalonePatternExtractor(dspy.Module)`. This module will chain multiple `dspy` calls to populate all the fields defined in the `PatternSignature`.

- [ ] **Update Main Script:** Modify `main.py` to call this new module and save the full extracted pattern to `/outputs/[paper_id]/pattern.json`.

- [ ] **Validation:**
    - Run the full extraction on 2-3 papers from the collection.
    - Review the output JSON files in `/outputs/[paper_id]/`. Is the structure correct?
    - Copy the `Structure` field's value into a Mermaid.js online editor to check if the diagram is valid and makes sense.

### Iteration 4: Setup the Knowledge Base

**Goal:** Persist extracted patterns and paper sections in a local vector database to enable retrieval.

- [ ] **Implement Knowledge Base Manager:** In `/src/knowledge_base.py`, create a class to manage the ChromaDB instance. It should handle:
    - Initializing a persistent ChromaDB client (`chromadb.PersistentClient(path="./db")`).
    - Creating collections for "sections" and "patterns".
    - A method `add_paper(paper_id, sections)` to generate embeddings and store paper sections.
    - A method `add_pattern(paper_id, pattern_data)` to store an extracted pattern.

- [ ] **Batch Ingestion Script:** Create a new script, `ingest.py`, that:
    1. Iterates through all subdirectories in the `/papers` directory.
    2. For each paper ID, runs the `StandalonePatternExtractor` from Iteration 3.
    3. Uses the `KnowledgeBaseManager` to add the paper's sections (from `/outputs/[paper_id]/sections.json`) and the extracted pattern (from `/outputs/[paper_id]/pattern.json`) to ChromaDB.

- [ ] **Validation:**
    - Run `ingest.py`.
    - Use the `KnowledgeBaseManager` in a separate script or notebook to query the database.
    - Verify that the collections are populated and you can retrieve items by ID.

### Iteration 5: RAG-Powered Extraction

**Goal:** Use the patterns in the knowledge base to improve extraction quality on a new, unseen paper.

- [ ] **Configure DSPy Retrieval:** Set up a `dspy.ColBERTv2` or similar retrieval model within `dspy` that connects to your ChromaDB instance. This links `dspy`'s retrieval system to your data.

- [ ] **Create RAG Module:** In `modules.py`, create a `RAGPatternExtractor(dspy.Module)`.
    - It will use `dspy.Retrieve` to find the top 2-3 most similar patterns from the knowledge base.
    - It will then use a `dspy.ChainOfThought` call whose signature accepts this `context` along with the `paper_section` to guide the final extraction.

- [ ] **Update Main Script:** Add logic to `main.py` to handle a new paper:
    1. Take a path to a *new* paper's directory (e.g., `/papers/new_paper_id/`) as input.
    2. Run the `RAGPatternExtractor` on it.
    3. Print the RAG-generated pattern and also the result from the old `StandalonePatternExtractor` for comparison.

- [ ] **Validation:**
    - Find a new paper that is similar to the initial collection but is not in the database.
    - Run the comparison script.
    - Compare the two outputs. The RAG-based output should ideally be more consistent or accurate. Check if the "Based On" field correctly references concepts from the retrieved patterns.
