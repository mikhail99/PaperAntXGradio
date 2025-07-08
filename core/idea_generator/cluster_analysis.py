import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

# --- Path Setup ---
# Add project root to sys.path to allow for core imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from core.collections_manager import CollectionsManager

# --- Configuration ---
COLLECTION_NAME = "HuggingFaceDailyPapers"
NUM_CLUSTERS = 30  # You can adjust this number
SAMPLES_PER_CLUSTER = 5 # Number of article titles to print for each cluster

def perform_clustering():
    """
    Fetches all embeddings from the specified collection, performs K-Means clustering,
    and prints the results.
    """
    print("Initializing CollectionsManager...")
    manager = CollectionsManager()

    print(f"Loading collection '{COLLECTION_NAME}'...")
    collection = manager.get_collection_by_name(COLLECTION_NAME)

    if not collection or not collection.articles:
        print("Collection not found or is empty.")
        return

    all_articles = list(collection.articles.values())
    article_ids = [article.id for article in all_articles]
    
    print(f"Fetching embeddings for {len(article_ids)} articles. This may take a moment...")
    embeddings_map = manager.get_embeddings_for_articles(COLLECTION_NAME, article_ids)

    if not embeddings_map:
        print("Could not retrieve embeddings. Aborting.")
        return
        
    # Ensure the order of embeddings matches the order of articles
    embedding_list = [embeddings_map[article_id] for article_id in article_ids if article_id in embeddings_map]
    
    # Also create a filtered list of articles that have embeddings
    articles_with_embeddings = [article for article in all_articles if article.id in embeddings_map]

    if not embedding_list:
        print("No valid embeddings found to cluster.")
        return

    # Convert to a numpy array for scikit-learn
    X = np.array(embedding_list)
    print(f"Clustering {X.shape[0]} embeddings into {NUM_CLUSTERS} clusters...")

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
    kmeans.fit(X)
    
    # Get the cluster label for each article
    labels = kmeans.labels_

    # Group articles by their assigned cluster
    clusters = defaultdict(list)
    for article, label in zip(articles_with_embeddings, labels):
        clusters[label].append(article)

    print("\n--- Clustering Results ---")
    for i in range(NUM_CLUSTERS):
        print(f"\n--- Cluster {i+1} ({len(clusters[i])} articles) ---")
        # Print a sample of articles from the cluster
        for article in clusters[i][:SAMPLES_PER_CLUSTER]:
            print(f"  - {article.title}")
    print("\n--------------------------")

if __name__ == "__main__":
    try:
        perform_clustering()
    except ImportError:
        print("\nError: Missing required libraries.")
        print("Please install them by running: pip install scikit-learn numpy")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}") 