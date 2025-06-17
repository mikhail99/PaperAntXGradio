import os
import sys
import json
from datetime import datetime
from tqdm import tqdm

# Ensure project root is in sys.path for core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.collections_manager import CollectionsManager
from core.data_models import Article

# --- Configuration ---
INPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "0_library_crawler_output", "huggingface_papers_enhanced.json")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "1_library_chroma_db_output")
COLLECTION_NAME = "HuggingFaceDailyPapers"




def get_paper_entries_from_json(json_path):
    """Loads paper entries from the enhanced JSON file."""
    if not os.path.exists(json_path):
        print(f"Error: Input JSON file not found at {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_or_get_collection(manager: CollectionsManager, name: str, description: str):
    """
    Creates a collection if it doesn't exist, or gets the existing one.
    Adds all unique categories from the JSON as tags to this collection.
    """
    existing_collections = manager.get_all_collections()
    for coll in existing_collections:
        if coll.name == name:
            print(f"Collection '{name}' already exists with ID: {coll.id}")
            collection = coll
            break
    else:
        collection = manager.create_collection(name, description)
        print(f"Created new collection '{name}' with ID: {collection.id}")

    # Add all unique categories from JSON as tags to the collection
    all_paper_entries = get_paper_entries_from_json(INPUT_JSON_PATH)
    all_categories = set()
    for paper_data in all_paper_entries:
        categories = paper_data.get("categories", [])
        if isinstance(categories, list):
            all_categories.update(categories)
        elif isinstance(categories, str):
            all_categories.add(categories)

    if collection:
        loaded_collection = manager.get_collection(collection.id)
        if loaded_collection:
            existing_tags = set(tag.name for tag in loaded_collection.tags.values())
            new_tags_to_add = all_categories - existing_tags
            if new_tags_to_add:
                tags_str = ", ".join(new_tags_to_add)
                try:
                    manager.parse_and_add_tags(loaded_collection, tags_str)
                    print(f"Added {len(new_tags_to_add)} new unique categories as tags to collection '{name}'")
                except Exception as e:
                    print(f"Warning: Could not add tags to collection '{name}': {str(e)}")
            else:
                print(f"Collection '{name}' already has all relevant category tags.")
        else:
            print(f"Warning: Could not load collection {collection.id} to add tags.")
    return collection


def add_papers_to_collection(manager: CollectionsManager, collection_id: str):
    """Adds papers from the enhanced JSON to the specified ChromaDB collection."""
    collection = manager.get_collection(collection_id)
    if not collection:
        print(f"Collection with ID {collection_id} not found!")
        return

    papers_data = get_paper_entries_from_json(INPUT_JSON_PATH)
    articles_to_add = []
    skipped_count = 0
    existing_article_ids = {article_id for article_id in collection.articles}

    for paper_data in papers_data:
        arxiv_id = paper_data.get("arxiv_id")
        if not arxiv_id:
            print("Skipping paper with no arxiv_id.")
            skipped_count += 1
            continue

        if arxiv_id in existing_article_ids:
            skipped_count += 1
            continue
        
        # Parse publication date, providing a default if missing or invalid
        pub_date_str = paper_data.get("published")
        try:
            publication_date = datetime.fromisoformat(pub_date_str) if pub_date_str else datetime.now()
        except (ValueError, TypeError):
            publication_date = datetime.now()

        article = Article(
            id=arxiv_id,
            title=paper_data.get("title", "N/A"),
            authors=paper_data.get("authors", []),
            abstract=paper_data.get("abstract", ""),
            publication_date=publication_date,
            tags=paper_data.get("categories", []),
            favorite=False,
            rating=None,
            citation_count=paper_data.get("votes", 0), # Using 'votes' as citation_count
            notes=paper_data.get("comment") or "", # Using 'comment' field for notes
        )
        articles_to_add.append(article)

    if articles_to_add:
        print(f"Adding {len(articles_to_add)} new papers to collection '{collection.name}'...")
        for article_obj in tqdm(articles_to_add, desc="Adding papers to collection"):
            try:
                manager.add_article(collection_id, article_obj)
            except Exception as e:
                print(f"Failed to add article {article_obj.id} ({article_obj.title}): {e}")
        print(f"Finished adding {len(articles_to_add)} new papers.")
    else:
        print("No new papers to add (all papers from JSON already exist in the collection).")
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} papers that already exist in the collection.")


def main():
    """Main function to populate ChromaDB with papers from the enhanced JSON file."""
    print(f"Ensuring ChromaDB output directory exists at: {CHROMA_DB_DIR}")
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    

    manager = CollectionsManager(persist_directory=CHROMA_DB_DIR)

    collection = create_or_get_collection(
        manager,
        COLLECTION_NAME,
        "Collection of AI papers from Hugging Face, enhanced with arXiv metadata."
    )

    if collection:
        add_papers_to_collection(manager, collection.id)
    else:
        print(f"Could not create or get collection '{COLLECTION_NAME}'. Aborting.")
    
    print("Finished processing papers for ChromaDB.")

if __name__ == "__main__":
    main() 