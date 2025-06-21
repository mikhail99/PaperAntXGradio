# Placeholder for article management logic (CRUD, rating, tagging, notes)
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid

from .data_models import Article, Collection
from .collections_manager import CollectionsManager

class ArticleManager:
    def __init__(self, collections_manager: CollectionsManager) -> None:
        """Initialize ArticleManager with a reference to the CollectionsManager"""
        self.collections_manager = collections_manager
        
    def create_article(self, collection_name: str, title: str, authors: List[str], 
                      abstract: str, publication_date: Optional[Union[datetime, str]] = None,
                      tags: Optional[List[str]] = None) -> Optional[Article]:
        """Create a new article and add it to a collection"""
        # Get the collection
        collection = self.collections_manager.get_collection(collection_name)
        if not collection:
            return None
            
        # Create the article
        article_id = str(uuid.uuid4())
        article = Article(
            id=article_id,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=publication_date or datetime.now(),
            tags=tags or []
        )
        
        # Add to collection
        success = self.collections_manager.add_article(collection_name, article)
        if not success:
            return None
            
        return article
        
    def get_article(self, collection_name: str, article_id: str) -> Optional[Article]:
        """Get an article from a collection"""
        collection = self.collections_manager.get_collection(collection_name)
        if not collection or article_id not in collection.articles:
            return None
            
        return collection.articles[article_id]
        
    def update_article(self, collection_name: str, article: Article) -> bool:
        """Update an article in a collection"""
        return self.collections_manager.update_article(collection_name, article)
        
    def delete_article(self, collection_name: str, article_id: str) -> bool:
        """Delete an article from a collection"""
        return self.collections_manager.delete_article(collection_name, article_id)
        
    def rate_article(self, collection_name: str, article_id: str, rating: str) -> bool:
        """Rate an article (accept, reject, favorite)"""
        article = self.get_article(collection_name, article_id)
        if not article:
            return False
            
        article.rating = rating
        return self.update_article(collection_name, article)
        
    def add_tags(self, collection_name: str, article_id: str, tags: List[str]) -> bool:
        """Add tags to an article"""
        article = self.get_article(collection_name, article_id)
        if not article:
            return False
            
        # Add new tags (avoid duplicates)
        for tag in tags:
            if tag not in article.tags:
                article.tags.append(tag)
                
        return self.update_article(collection_name, article)
        
    def remove_tags(self, collection_name: str, article_id: str, tags: List[str]) -> bool:
        """Remove tags from an article"""
        article = self.get_article(collection_name, article_id)
        if not article:
            return False
            
        # Remove tags
        article.tags = [t for t in article.tags if t not in tags]
        return self.update_article(collection_name, article)
        
    def search_articles(self, collection_name: str, query: str, limit: int = 10) -> List[Article]:
        """Search for articles by semantic similarity"""
        return self.collections_manager.search_articles(collection_name, query, limit) 