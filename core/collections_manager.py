from typing import Dict, List, Optional, Any, cast
import uuid
import os
from pathlib import Path
from datetime import datetime
import json

from .data_models import Collection, Tag, Article
from .chroma_service import ChromaService

# Get the directory of the current file (e.g., .../core)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level and join with the data directory
SOURCE_CHROMA_DB_DIR = os.path.join(current_dir, "..", "data", "1_library_chroma_db_output", "chroma_db")

# --- IMPORTANT: Normalize the path ---
# This resolves the ".." and gives you a clean, absolute path
SOURCE_CHROMA_DB_DIR = os.path.normpath(SOURCE_CHROMA_DB_DIR)

print(SOURCE_CHROMA_DB_DIR)
# This will now print: C:\Users\mihai\Documents\Projects\PaperAntXGradio\data\1_library_chroma_db_output

class CollectionsManager:
    def __init__(self, persist_directory: str = SOURCE_CHROMA_DB_DIR) -> None:
        """Initialize CollectionsManager with ChromaDB backend"""
        self.chroma_service = ChromaService(persist_directory=persist_directory)
        self.collections: Dict[str, Collection] = {}
        self.load_collections()

    def article_to_metadata(self, article: Article) -> Dict[str, Any]:
        """Convert Article to ChromaDB metadata"""
        pub_date = (
            article.publication_date.isoformat()
            if isinstance(article.publication_date, datetime)
            else article.publication_date
        )
        return {
            "title": article.title,
            "authors": article.authors,
            "publication_date": pub_date,
            "tags": article.tags,
            "favorite": article.favorite,
            "rating": article.rating,
            "citation_count": article.citation_count,
            "notes": article.notes,
            "abstract": article.abstract,
            "url": article.url,
        }

    def metadata_to_article(self, article_id: str, metadata: Dict[str, Any], document: str = "") -> Article:
        """Convert ChromaDB metadata to Article"""
        # Extract title and abstract from document if not in metadata
        if document and "title" not in metadata:
            # Simple heuristic: first line is title, rest is abstract
            parts = document.split(" ", 1)
            title = parts[0] if len(parts) > 0 else "Unknown Title"
            abstract = parts[1] if len(parts) > 1 else ""
        else:
            title = metadata.get("title", "Unknown Title")
            abstract = metadata.get("abstract", "")

        return Article(
            id=article_id,
            title=title,
            authors=metadata.get("authors", []),
            abstract=abstract,
            publication_date=metadata.get("publication_date", datetime.now().isoformat()),
            url=metadata.get("url"),
            tags=metadata.get("tags", []),
            favorite=metadata.get("favorite", False),
            rating=metadata.get("rating"),
            citation_count=metadata.get("citation_count", 0),
            notes=metadata.get("notes", ""),
        )

    def collection_to_metadata(self, collection: Collection) -> Dict[str, Any]:
        """Convert Collection to ChromaDB metadata"""
        # Convert tags to a serializable format
        tags_data = {}
        for tag_id, tag in collection.tags.items():
            tags_data[tag_id] = {
                "name": tag.name,
                "color": tag.color,
                "parent_id": tag.parent_id,
            }

        return {
            "name": collection.name,
            "description": collection.description,
            "tags": json.dumps(tags_data),
            "archived": collection.archived,
        }

    def metadata_to_collection(self, metadata: Dict[str, Any]) -> Collection:
        """Convert ChromaDB metadata to Collection"""
        # Deserialize tags if they are a JSON string
        tags_str = metadata.get("tags", "{}")
        tags_data = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
        
        # Create tags dictionary
        tags: Dict[str, Tag] = {}
        for tag_id, tag_data in tags_data.items():
            tags[tag_id] = Tag(
                id=tag_id,
                name=tag_data["name"],
                color=tag_data.get("color", "#1890ff"),
                parent_id=tag_data.get("parent_id"),
            )

        return Collection(
            name=metadata.get("name", "Unnamed Collection"),
            description=metadata.get("description", ""),
            tags=tags,
            archived=metadata.get("archived", False),
        )

    def load_collections(self) -> None:
        """Load all collections from ChromaDB"""
        # Get all collections from ChromaDB
        all_collections = self.chroma_service.client.list_collections()
        
        # Convert each ChromaDB collection to a PaperAnt Collection
        for chroma_collection in all_collections:
            collection_name = chroma_collection.name
            
            # Get collection metadata
            collection_obj = self.chroma_service.get_collection(collection_name)
            if not collection_obj:
                continue
                
            # Get deserialized collection metadata
            collection_metadata = self.chroma_service.get_collection_metadata(collection_name) or {}
            
            # Convert to Collection object
            collection = self.metadata_to_collection(collection_metadata)
            
            # Load articles from this collection
            articles_data = self.chroma_service.get_articles(collection_name)
            articles: Dict[str, Article] = {}
            
            for article_data in articles_data:
                article_id = article_data["id"]
                article = self.metadata_to_article(
                    article_id, 
                    article_data["metadata"], 
                    article_data.get("document", "")
                )
                articles[article_id] = article
                
            # Add articles to collection
            collection.articles = articles
            
            # Add to our in-memory cache
            self.collections[collection_name] = collection

    def save_collection(self, collection: Collection) -> bool:
        """Saves a collection's metadata to ChromaDB."""
        collection_name = collection.name
        collection_metadata = self.collection_to_metadata(collection)
        
        try:
            # Get the existing collection from ChromaDB
            chroma_collection = self.chroma_service.client.get_collection(name=collection_name)
            # Modify its metadata
            chroma_collection.modify(metadata=collection_metadata)
        except Exception as e:
            print(f"Error saving collection '{collection_name}': {e}")
            # Fallback to creating it if it truly doesn't exist, though get_collection should handle this.
            try:
                self.chroma_service.create_collection(
                    collection_name=collection_name,
                    metadata=collection_metadata
                )
            except Exception as e2:
                 print(f"Failed to create collection '{collection_name}' as a fallback: {e2}")
                 return False

        # Update in-memory cache
        self.collections[collection_name] = collection
        return True

    def create_collection(self, name: str, description: str) -> Collection:
        """Create a new collection"""
        
        collection = Collection(name=name, description=description)
        
        # Save to ChromaDB
        success = self.save_collection(collection)
        if not success:
            raise ValueError(f"Failed to create collection: {name}")
            
        return collection

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """Get a collection by ID"""
        # Try from in-memory cache first
        if collection_name in self.collections:
            return self.collections[collection_name]
            
        # If not in cache, try to get from ChromaDB
        chroma_collection = self.chroma_service.get_collection(collection_name)
        if not chroma_collection:
            return None
            
        # Convert to Collection object - get deserialized metadata
        collection_metadata = self.chroma_service.get_collection_metadata(collection_name) or {}
        collection = self.metadata_to_collection(collection_metadata)
        
        # Load articles from this collection
        articles_data = self.chroma_service.get_articles(collection_name)
        articles: Dict[str, Article] = {}
        
        for article_data in articles_data:
            article_id = article_data["id"]
            article = self.metadata_to_article(
                article_id, 
                article_data["metadata"], 
                article_data.get("document", "")
            )
            articles[article_id] = article
            
        # Add articles to collection
        collection.articles = articles
        
        # Add to our in-memory cache
        self.collections[collection_name] = collection
        return collection

    def get_collection_by_name(self, name: str) -> Optional[Collection]:
        """Finds and returns a collection by its name from the in-memory cache."""
        for collection in self.collections.values():
            if collection.name == name:
                return collection
        return None

    def update_collection(
        self, collection_name: str,  description: Optional[str] = None
    ) -> Optional[Collection]:
        """Update a collection"""
        collection = self.get_collection(collection_name)
        if not collection:
            return None


        if description is not None:
            collection.description = description

        # Save changes to ChromaDB
        success = self.save_collection(collection)

            
        return collection

    def archive_collection(self, collection_name: str, archived: bool = True) -> Optional[Collection]:
        """Archive or unarchive a collection"""
        collection = self.get_collection(collection_name)
        if not collection:
            return None

        collection.archived = archived
        
        # Save changes to ChromaDB
        success = self.save_collection(collection)
        if not success:
            raise ValueError(f"Failed to archive collection: {collection.name}")
            
        return collection

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        # Delete from ChromaDB
        success = self.chroma_service.delete_collection(collection_name)
        
        # Remove from in-memory cache
        if collection_name in self.collections:
            del self.collections[collection_name]
            
        return success

    def get_all_collections(self, include_archived: bool = False) -> List[Collection]:
        """Get all collections"""
        # Return from in-memory cache
        if include_archived:
            return list(self.collections.values())
        return [c for c in self.collections.values() if not c.archived]

    def parse_and_add_tags(self, collection: Collection, tags_str: str) -> Dict[str, Tag]:
        """Parse a comma-separated tag string and add Tag objects to the collection. Returns the tag dict."""
        tags = {}
        for tag_entry in [t.strip() for t in tags_str.split(",") if t.strip()]:
            # Support parent/child syntax
            if "/" in tag_entry:
                parent, child = tag_entry.split("/", 1)
                parent_tag = Tag(id=str(uuid.uuid4()), name=parent.strip())
                child_tag = Tag(id=str(uuid.uuid4()), name=child.strip(), parent_id=parent_tag.id)
                tags[parent_tag.id] = parent_tag
                tags[child_tag.id] = child_tag
            else:
                tag = Tag(id=str(uuid.uuid4()), name=tag_entry)
                tags[tag.id] = tag
        collection.tags = tags
        
        # Save changes to ChromaDB
        self.save_collection(collection)
        return tags
        
    def add_article(self, collection_name: str, article: Article) -> bool:
        """Add a single article to a collection. Consider using add_articles_batch for multiple articles."""
        return self.add_articles_batch(collection_name, [article])

    def add_articles_batch(self, collection_name: str, articles: List[Article]) -> bool:
        """Adds a batch of articles to a collection efficiently."""
        collection = self.get_collection_by_name(collection_name)
        if not collection:
            return False

        articles_to_add_to_db = []
        for article in articles:
            articles_to_add_to_db.append({
                "id": article.id,
                "title": article.title,
                "abstract": article.abstract,
                "metadata": self.article_to_metadata(article)
            })

        if not articles_to_add_to_db:
            return True # Nothing to add

        # The add_articles service expects a list of dictionaries
        success = self.chroma_service.add_articles(
            collection_name=collection_name,
            articles=articles_to_add_to_db
        )

        if not success:
            return False

        # Update in-memory cache
        for article in articles:
            collection.articles[article.id] = article
        self.collections[collection_name] = collection
        return True
        
    def update_article(self, collection_name: str, article: Article) -> bool:
        """Update an article in a collection"""
        collection = self.get_collection_by_name(collection_name)
        if not collection:
            return False
            
        # Update in ChromaDB
        document = f"{article.title} {article.abstract}"
        metadata = self.article_to_metadata(article)
        
        success = self.chroma_service.update_article(
            collection_name=collection_name,
            article_id=article.id,
            document=document,
            metadata=metadata
        )
        
        if not success:
            return False
            
        # Update in-memory cache
        collection.articles[article.id] = article
        self.collections[collection_name] = collection
        return True
        
    def delete_article(self, collection_name: str, article_id: str) -> bool:
        """Delete an article from a collection"""
        collection = self.get_collection(collection_name)
        if not collection:
            return False
            
        # Delete from ChromaDB
        success = self.chroma_service.delete_article(
            collection_name=collection_name,
            article_id=article_id
        )
        
        if not success:
            return False
            
        # Delete from in-memory cache
        if article_id in collection.articles:
            del collection.articles[article_id]
            self.collections[collection_name] = collection
        return True
        
    def search_articles(self, collection_name: str, query: str, limit: int = 10) -> List[Article]:
        """Search for articles in a collection by semantic similarity"""
        collection = self.get_collection(collection_name)
        if not collection:
            return []
            
        # Search in ChromaDB
        articles_data = self.chroma_service.search_articles(
            collection_name=collection_name,
            query=query,
            limit=limit
        )
        
        # Convert to Article objects
        articles = []
        for article_data in articles_data:
            article_id = article_data["id"]
            article = self.metadata_to_article(
                article_id, 
                article_data["metadata"], 
                article_data.get("document", "")
            )
            articles.append(article)
            
        return articles
