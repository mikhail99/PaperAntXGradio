import os
import sys
import asyncio
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paperqa import Docs
from core.collections_manager import CollectionsManager
from core.data_models import Article
from core.utils import get_local_llm_settings

# --- Configuration ---
LLM_MODEL = "ollama/gemma3:4b"
EMBEDDING_MODEL = "ollama/nomic-embed-text:latest"
SOURCE_CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "1_library_chroma_db_output")

# --- Helper Functions ---

def format_docname(article: Article) -> str:
    """Creates a clean, readable docname from an Article object."""
    title = article.title or "Unknown Title"
    
    # Format authors to "First Author et al." if more than one
    authors = article.authors
    if authors and len(authors) > 1:
        author_str = f"{authors[0]} et al."
    elif authors:
        author_str = authors[0]
    else:
        author_str = "Unknown Author"
        
    year = article.publication_date.year if article.publication_date else "Unknown Year"
    return f"{title} ({author_str}, {year})"

# --- Main Logic ---

async def build_cache_from_collection(collection_name: str):
    """
    Builds a PaperQA cache directly from a collection's ChromaDB and its PDFs.
    """
    print(f"--- Building PaperQA Cache for collection: '{collection_name}' ---")
    
    # 1. Define paths based on the collection name
    base_collection_path = Path("data/collections") / collection_name
    pdf_folder = base_collection_path / "pdfs"
    # The cache is now a single pickle file
    cache_file_path = base_collection_path / "paperqa_cache.pkl"



    # Create the output directory if it doesn't exist
    cache_file_path.parent.mkdir(parents=True, exist_ok=True)


    chromadb_manager = CollectionsManager(persist_directory=SOURCE_CHROMA_DB_DIR)
    chrimadb_collection = chromadb_manager.get_collection_by_name(collection_name)    
    articles: List[Article] = list(chrimadb_collection.articles.values())
    if not articles:
        print("No articles found in the collection. Nothing to cache.")
        return
    
    print(f"Found {len(articles)} articles in the collection.")

    # 3. Initialize PaperQA Docs instance and configure LLM settings
    llm_settings = get_local_llm_settings(LLM_MODEL, EMBEDDING_MODEL)
    docs = Docs() # Docs is initialized with no arguments

    # 4. Add documents to build the in-memory cache
    print("Building in-memory cache...")
    for article in tqdm(articles, desc="Building PaperQA Cache"):
        pdf_path = pdf_folder / f"{article.id}.pdf"
        if not pdf_path.exists():
            print(f"\nWarning: PDF for article '{article.title}' not found at {pdf_path}. Skipping.")
            continue
            
        docname = format_docname(article)
        citation = format_docname(article)
        
        # The settings object carries all configuration.
        await docs.aadd(path=str(pdf_path), docname=docname, citation=citation, settings=llm_settings)

    # 5. Save the populated Docs object to a pickle file
    print(f"\nSaving populated cache to: {cache_file_path}")
    try:
        with open(cache_file_path, "wb") as f:
            pickle.dump(docs, f)
        print("\n--- PaperQA cache build complete! ---")
        print(f"Cache for '{collection_name}' is saved and ready for use at: {cache_file_path}")
    except Exception as e:
        print(f"An error occurred while saving the pickle file: {e}")


if __name__ == "__main__":
    # Build a PaperQA cache for a curated collection using its ChromaDB for metadata.
    collection_name = "LLM_Reasoning_Agents"
    asyncio.run(build_cache_from_collection(collection_name)) 