#!/usr/bin/env python3
from core.collections_manager import CollectionsManager
from core.article_manager import ArticleManager
from datetime import datetime

# Initialize the managers
manager = CollectionsManager(persist_directory="data/chroma_db_store")
article_manager = ArticleManager(manager)

# Get collections
collections = manager.get_all_collections()
print(f"Found {len(collections)} collections")

if not collections:
    print("No collections found")
    exit(0)

# Print information about each collection and its articles
for i, collection in enumerate(collections):
    print(f"\nCollection {i+1}: {collection.name} (ID: {collection.id})")
    print(f"Description: {collection.description}")
    print(f"Articles: {len(collection.articles)}")
    
    # Print details of each article
    for j, (article_id, article) in enumerate(collection.articles.items()):
        print(f"\n  Article {j+1}: {article.title}")
        print(f"  ID: {article_id}")
        print(f"  Authors: {', '.join(article.authors)}")
        print(f"  Abstract: {article.abstract[:100] + '...' if len(article.abstract) > 100 else article.abstract}")
        print(f"  Notes: {article.notes[:50] + '...' if article.notes and len(article.notes) > 50 else article.notes or 'None'}")
        
        # Save a test note to the article if it has no notes
        if not article.notes:
            test_note = f"Test note for article {article.title} written at {datetime.now()}"
            article.notes = test_note
            success = article_manager.update_article(collection.id, article)
            print(f"  Added test note: {success}")

print("\nDebug complete") 