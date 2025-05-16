#!/usr/bin/env python3
import os
import uuid
from datetime import datetime, timedelta
import random

from core.data_models import Collection, Article
from core.collections_manager import CollectionsManager

# Sample data
MOCK_PAPER_TITLES = [
    "Advances in Large Language Models: A Comparative Study",
    "Quantum Computing: Challenges and Opportunities",
    "Deep Learning for Computer Vision: A Comprehensive Review",
    "Reinforcement Learning in Robotics Applications",
    "The Future of Natural Language Processing",
    "Generative Adversarial Networks: Recent Developments",
    "Ethics in Artificial Intelligence: Current Perspectives",
    "Federated Learning for Privacy-Preserving AI",
    "Explainable AI: Methods and Applications",
    "Graph Neural Networks for Knowledge Representation"
]

MOCK_AUTHORS = [
    ["Zhang, L.", "Smith, J.", "Kumar, A."],
    ["Johnson, R.", "Williams, M."],
    ["Garcia, C.", "Brown, T.", "Davis, S."],
    ["Martinez, E.", "Anderson, K."],
    ["Wilson, P.", "Lee, J.", "Patel, R."],
    ["Taylor, S.", "White, D."],
    ["Harris, M.", "Clark, B.", "Lewis, A."],
    ["Miller, J.", "Moore, R."],
    ["Jackson, T.", "Thompson, L.", "Walker, N."],
    ["Young, K.", "Allen, C."]
]

MOCK_ABSTRACTS = [
    "This paper presents a comparative study of recent advancements in large language models, focusing on their capabilities, limitations, and potential applications across various domains.",
    "We explore the current state of quantum computing, discussing technical challenges, recent breakthroughs, and promising applications that could revolutionize computing in the near future.",
    "Our review examines the evolution of deep learning techniques in computer vision, highlighting key architectures, datasets, and benchmarks that have driven progress in this field.",
    "This research investigates reinforcement learning approaches for robotic control and decision-making, with experiments in both simulated and real-world environments.",
    "We analyze emerging trends in natural language processing, including multilingual models, few-shot learning, and ethical considerations for deployment at scale.",
    "This paper surveys recent innovations in generative adversarial networks, focusing on applications in image synthesis, style transfer, and domain adaptation.",
    "Our analysis explores ethical frameworks for artificial intelligence development and deployment, with case studies highlighting current challenges and proposed solutions.",
    "We present advances in federated learning techniques that enable collaborative model training while preserving data privacy across distributed systems.",
    "This research examines methods for making AI systems more interpretable and transparent, with applications in healthcare, finance, and autonomous vehicles.",
    "Our work explores how graph neural networks can effectively represent and reason with complex relational knowledge structures."
]

MOCK_TAGS = [
    ["large language models", "NLP", "transformers", "GPT", "deep learning"],
    ["quantum computing", "qubits", "quantum algorithms", "quantum supremacy"],
    ["computer vision", "CNN", "object detection", "image classification", "deep learning"],
    ["reinforcement learning", "robotics", "RL", "control systems", "autonomous agents"],
    ["NLP", "language models", "multilingual", "few-shot learning"],
    ["GAN", "generative models", "image synthesis", "style transfer", "deep learning"],
    ["AI ethics", "responsible AI", "fairness", "accountability", "transparency"],
    ["federated learning", "privacy", "distributed systems", "security"],
    ["XAI", "explainable AI", "interpretability", "model transparency", "healthcare"],
    ["GNN", "graph neural networks", "knowledge graphs", "representation learning"]
]

def create_collection_if_not_exists(manager, name, description):
    """Create a collection if it doesn't already exist"""
    # Check if collection already exists
    collections = manager.get_all_collections()
    for collection in collections:
        if collection.name == name:
            print(f"Collection '{name}' already exists with ID: {collection.id}")
            
            # Optional: Update the collection tags without re-creating it
            all_tags = set()
            for tag_list in MOCK_TAGS:
                all_tags.update(tag_list)
            
            # Check if the collection already has these tags
            existing_tags = set(tag.name for tag in collection.tags.values())
            new_tags = all_tags - existing_tags
            
            if new_tags:
                tags_str = ", ".join(new_tags)
                try:
                    manager.parse_and_add_tags(collection, tags_str)
                    print(f"Added {len(new_tags)} new tags to existing collection")
                except Exception as e:
                    print(f"Warning: Could not add tags to collection: {str(e)}")
            else:
                print("Collection already has all the mock tags")
                
            return collection
    
    # Create new collection
    collection = manager.create_collection(name, description)
    print(f"Created new collection '{name}' with ID: {collection.id}")
    
    # Add tags to the collection - we do this in a separate step to avoid the error
    try:
        # Get the collection fresh from the manager to ensure it's properly loaded
        collection = manager.get_collection(collection.id)
        
        all_tags = set()
        for tag_list in MOCK_TAGS:
            all_tags.update(tag_list)
        
        tags_str = ", ".join(all_tags)
        manager.parse_and_add_tags(collection, tags_str)
        print(f"Added {len(all_tags)} unique tags to collection")
    except Exception as e:
        print(f"Warning: Could not add tags to collection: {str(e)}")
    
    return collection

def add_mock_articles(manager, collection_id, num_articles=10):
    """Add mock articles to the specified collection"""
    # Get the collection fresh from the manager to ensure it's properly loaded
    collection = manager.get_collection(collection_id)
    if not collection:
        print(f"Collection with ID {collection_id} not found!")
        return
        
    # Check if articles already exist
    existing_article_count = len(collection.articles)
    if existing_article_count > 0:
        print(f"Collection already has {existing_article_count} articles")
        # Optional: skip adding more articles if some already exist
        if existing_article_count >= num_articles:
            print("Skipping article creation as collection already has enough articles")
            return
            
    print(f"Adding {num_articles} mock articles to collection...")
    for i in range(num_articles):
        # Generate article data
        title = MOCK_PAPER_TITLES[i % len(MOCK_PAPER_TITLES)]
        authors = MOCK_AUTHORS[i % len(MOCK_AUTHORS)]
        abstract = MOCK_ABSTRACTS[i % len(MOCK_ABSTRACTS)]
        tags = MOCK_TAGS[i % len(MOCK_TAGS)]
        
        # Generate random publication date within the last 2 years
        days_ago = random.randint(0, 730)  # 0-730 days (about 2 years)
        pub_date = datetime.now() - timedelta(days=days_ago)
        
        # Create the article
        article = Article(
            id=str(uuid.uuid4()),
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            tags=tags,
            favorite=random.random() > 0.8,  # 20% chance of being favorite
            rating=random.choice([None, "accept", "reject", "favorite"]),
            citation_count=random.randint(0, 500)
        )
        
        # Add to collection
        success = manager.add_article(collection_id, article)
        if success:
            print(f"Added article: {title}")
        else:
            print(f"Failed to add article: {title}")

def main():
    # Ensure data directory exists
    os.makedirs("data/chroma_db_store", exist_ok=True)
    
    # Initialize collection manager
    manager = CollectionsManager(persist_directory="data/chroma_db_store")
    
    # Use a different collection name to avoid conflicts
    collection_name = "Collection 1"
    collection = create_collection_if_not_exists(
        manager,
        collection_name,
        "A collection of research papers on AI and machine learning topics"
    )
    
    # Add mock articles
    add_mock_articles(manager, collection.id)
    
    print("Finished populating collection with mock data")

if __name__ == "__main__":
    main() 