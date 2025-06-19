import os
import sys
import json
from datetime import datetime
from tqdm import tqdm

# Ensure project root is in sys.path for core imports
# --- Path Configuration ---
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from core.collections_manager import CollectionsManager
from core.data_models import Article

# --- Configuration ---
INPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "0_library_crawler_output", "huggingface_papers_enhanced.json")
COLLECTION_NAME = "HuggingFaceDailyPapers"




def get_paper_entries_from_json(json_path: str) -> list[dict]:
    """Reads and returns all paper entries from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _get_all_categories_from_json(json_path: str) -> set[str]:
    """Reads all unique categories from the paper entries in the JSON file."""
    all_paper_entries = get_paper_entries_from_json(json_path)
    all_categories = set()
    for paper_data in all_paper_entries:
        categories = paper_data.get("categories", [])
        if isinstance(categories, list):
            all_categories.update(categories)
        elif isinstance(categories, str):
            all_categories.add(categories)
    return all_categories

def create_or_get_collection(manager: CollectionsManager, name: str, description: str):
    """
    Creates a collection if it doesn't exist, or gets the existing one.
    It then ensures that all unique categories from the input JSON are added as tags.
    """
    # 1. Get or create the collection
    collection = manager.get_collection_by_name(name=name)
    if not collection:
        collection = manager.create_collection(name, description)
        print(f"Created new collection: '{name}'")
    else:
        print(f"Collection '{name}' already exists.")

    # 2. Get all required categories from the source JSON
    all_categories = _get_all_categories_from_json(INPUT_JSON_PATH)

    # 3. Determine which new tags need to be added
    existing_tags = set(tag.name for tag in collection.tags.values())
    new_tags_to_add = all_categories - existing_tags

    if not new_tags_to_add:
        print(f"Collection '{name}' already has all relevant category tags.")
        return collection

    # 4. Add the new tags
    print(f"Found {len(new_tags_to_add)} new tags to add to the collection.")
    tags_str = ", ".join(new_tags_to_add)
    try:
        manager.parse_and_add_tags(collection, tags_str)
        print(f"Successfully added new tags to '{name}'.")
    except Exception as e:
        print(f"Warning: Could not add tags to collection '{name}': {e}")
    
    return collection


def _parse_paper_data_to_article(paper_data: dict) -> Article | None:
    """Parses a dictionary of paper data into an Article object."""
    arxiv_id = paper_data.get("arxiv_id")
    if not arxiv_id:
        print("Skipping paper with no arxiv_id.")
        return None

    # Parse publication date, providing a default if missing or invalid
    pub_date_str = paper_data.get("published")
    try:
        publication_date = datetime.fromisoformat(pub_date_str) if pub_date_str else datetime.now()
    except (ValueError, TypeError):
        publication_date = datetime.now()

    return Article(
        id=arxiv_id,
        title=paper_data.get("title", "N/A"),
        authors=paper_data.get("authors", []),
        abstract=paper_data.get("abstract", ""),
        publication_date=publication_date,
        tags=paper_data.get("categories", []),
        favorite=False,
        rating=None,
        citation_count=paper_data.get("votes", 0),
        notes=paper_data.get("comment") or "",
    )


def add_papers_to_collection(manager: CollectionsManager, collection_name: str, limit: int | None = 100):
    """
    Adds new papers from the source JSON to the specified collection in a single batch.
    """
    collection = manager.get_collection_by_name(collection_name)
    if not collection:
        print(f"Collection '{collection_name}' not found.")
        return

    papers_data = get_paper_entries_from_json(INPUT_JSON_PATH)
    existing_article_ids = set(collection.articles.keys())
    articles_to_add = []

    print("Parsing paper data and filtering existing articles...")
    for paper_data in tqdm(papers_data, desc="Parsing papers"):
        if paper_data.get("arxiv_id") in existing_article_ids:
            continue
        
        article = _parse_paper_data_to_article(paper_data)
        if article:
            articles_to_add.append(article)
    
    # --- Deduplicate the list of new articles before adding ---
    seen_ids = set()
    unique_articles_to_add = []
    for article in articles_to_add:
        if article.id not in seen_ids:
            unique_articles_to_add.append(article)
            seen_ids.add(article.id)
    
    if len(unique_articles_to_add) < len(articles_to_add):
        print(f"Removed {len(articles_to_add) - len(unique_articles_to_add)} duplicate articles from the new batch.")
    
    articles_to_add = unique_articles_to_add
    # --- End Deduplication ---

    if limit is not None:
        articles_to_add = articles_to_add[:limit]
    
    if not articles_to_add:
        print("No new papers to add to the collection.")
        return

    # --- Chunking for large batches ---
    chunk_size = 4000  # Keep well under ChromaDB's max batch size of 5461
    total_added = 0
    
    print(f"\nPreparing to add {len(articles_to_add)} new papers in chunks of {chunk_size}...")

    for i in tqdm(range(0, len(articles_to_add), chunk_size), desc="Adding batches to DB"):
        chunk = articles_to_add[i:i + chunk_size]
        try:
            success = manager.add_articles_batch(collection_name, chunk)
            if success:
                total_added += len(chunk)
                print(f"Successfully added chunk of {len(chunk)}. Total added so far: {total_added}")
            else:
                print(f"Failed to add a chunk of {len(chunk)} papers. Aborting.")
                break  # Stop if a chunk fails
        except Exception as e:
            print(f"An exception occurred during batch add for a chunk: {e}")
            break  # Stop if a chunk fails
    
    print(f"\nFinished batch processing. Total articles successfully added: {total_added}")


def main():
    """Main function to populate ChromaDB with papers from the enhanced JSON file."""

    
    manager = CollectionsManager()

    
    collection = create_or_get_collection(
        manager,
        COLLECTION_NAME,
        "Collection of AI papers from Hugging Face, enhanced with arXiv metadata."
    )

    if collection:
        add_papers_to_collection(manager, collection.name, limit=None)
    else:
        print(f"Could not create or get collection '{COLLECTION_NAME}'. Aborting.")
   
 
    print("Finished processing papers for ChromaDB.")


    
    print("\n--- Debugging Information ---")
    all_collections_summary = manager.get_all_collections()
    if not all_collections_summary:
        print("No collections found.")
    else:
        print(f"Found {len(all_collections_summary)} collections:")
        for coll_summary in all_collections_summary:
            collection = manager.get_collection_by_name(coll_summary.name)
            if collection:
                num_articles = len(collection.articles)
                print(f"- Collection: '{collection.name}' contains {num_articles} articles.")
            else:
                 print(f"- Collection: '{coll_summary.name}' could not be loaded.")
    print("---------------------------\n")


if __name__ == "__main__":
    main() 