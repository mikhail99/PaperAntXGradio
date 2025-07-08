import os
import sys

# Add project root to sys.path to allow for core imports
# This is navigating up from core/idea_generator/test.py to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from core.collections_manager import CollectionsManager

def main():
    """
    This script tests reading data from the ChromaDB library collection.
    """
    print("Attempting to connect to ChromaDB and load collections...")
    
    # CollectionsManager finds the DB path relative to its own file location,
    # so it should work seamlessly.
    
    manager = CollectionsManager()
    collection_name = "HuggingFaceDailyPapers"
    collection = manager.get_collection_by_name(collection_name)

    print(f"\n✅ Successfully loaded collection: '{collection.name}'")
    num_articles = len(collection.articles)
    print(f"   Number of articles in collection: {num_articles}")

    if num_articles > 0:
        print("\n--- First 5 Articles in Collection ---")
        articles_to_sample = list(collection.articles.values())[:5]
        for i, article in enumerate(articles_to_sample):
            print(f"  {i+1}. ID: {article.id}, Title: {article.title}")
        
        # Get embeddings for the sampled articles
        article_ids_to_fetch = [article.id for article in articles_to_sample]
        print("\n--- Fetching embeddings for sampled articles ---")
        try:
            embeddings = manager.get_embeddings_for_articles(collection_name, article_ids_to_fetch)
            if embeddings:
                print(f"Successfully retrieved {len(embeddings)} embeddings.")
                for article_id, embedding in embeddings.items():
                    # The embedding can be very long, so just print its shape/length
                    print(f"  - Embedding for article {article_id}: Shape/Length = {len(embedding)}")
            else:
                print("Could not retrieve embeddings for the articles.")
        except Exception as e:
            print(f"An error occurred while fetching embeddings: {e}")

        print("--------------------------------------\n")




if __name__ == "__main__":
    main()

