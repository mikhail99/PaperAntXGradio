# Placeholder for ChromaDB integration and vector search logic 

import chromadb
from chromadb.utils import embedding_functions
import os
import json
from typing import Dict, List, Optional, Any, Union, Mapping
from uuid import uuid4

class ChromaService:
    def __init__(self, persist_directory: str = "data/chroma_db_store") -> None:
        """Initialize ChromaDB client with persistence"""
        self.persist_directory = persist_directory
        # Create the directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client: Any = chromadb.PersistentClient(path=self.persist_directory)  # type: ignore
        
        # Use SentenceTransformer for embeddings (lightweight model)
        #self.embedding_function = None #embedding_functions.SentenceTransformerEmbeddingFunction(
        #    #model_name="all-MiniLM-L6-v2"
        #)
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, bool, int, float, None]]:
        """Convert metadata to ChromaDB-compatible format (simple types only)"""
        prepared_metadata: Dict[str, Union[str, bool, int, float, None]] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, bool, int, float)):
                prepared_metadata[key] = value
            elif value is None:
                # Skip None values
                continue
            else:
                # Serialize any complex types (like nested dicts) to JSON string
                prepared_metadata[key] = json.dumps(value)
        return prepared_metadata
    
    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON-serialized strings back to Python objects"""
        deserialized = {}
        for key, value in metadata.items():
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    deserialized[key] = json.loads(value)
                except json.JSONDecodeError:
                    # If not valid JSON, keep as string
                    deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict[str, Any]] = None) :
        """Create a new ChromaDB collection for a PaperAnt collection"""
        try:
            prepared_metadata = self._prepare_metadata(metadata or {})
            return self.client.create_collection(
                name=collection_name,
                #embedding_function=self.embedding_function,
                metadata=prepared_metadata
            )
        except ValueError:
            # Collection might already exist
            collection = self.get_collection(collection_name)
            if collection and metadata:
                # Update metadata if collection exists
                prepared_metadata = self._prepare_metadata(metadata)
                collection.metadata = prepared_metadata
            return collection
    
    def get_collection(self, collection_name: str):
        """Get a ChromaDB collection by ID"""
        return self.client.get_collection(
                name=collection_name,
                #embedding_function=self.embedding_function
            )

    
    def get_collection_metadata(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get a collection's metadata, deserializing any JSON strings"""
        collection = self.get_collection(collection_name)
        if not collection:
            return None
        
        return self._deserialize_metadata(collection.metadata or {})
    
    def update_collection_metadata(self, collection_name: str, metadata: Dict[str, Any]) -> bool:
        """Update a collection's metadata"""
        raise NotImplementedError("Not implemented")
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a ChromaDB collection"""
        try:
            self.client.delete_collection(name=collection_name)
            return True
        except ValueError:
            # Collection doesn't exist
            return False
    
    def add_articles(self, collection_name: str, articles: List[Dict[str, Any]]) -> bool:
        """Add articles to a ChromaDB collection
        
        articles: List of dicts with keys: id, title, abstract, and metadata
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return False
        
        ids = [article["id"] for article in articles]
        documents = [f"{article['title']} {article['abstract']}" for article in articles]
        
        # Prepare metadata - ensure all values are simple types
        metadatas = []
        for article in articles:
            metadata = article.get("metadata", {})
            metadatas.append(self._prepare_metadata(metadata))
        
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Error adding articles to ChromaDB: {e}")
            return False
    
    def update_article(self, collection_name: str, article_id: str, document: str, metadata: Dict[str, Any]) -> bool:
        """Update an article in a ChromaDB collection"""
        collection = self.get_collection(collection_name)
        if not collection:
            return False
        
        prepared_metadata = self._prepare_metadata(metadata)
        
        try:
            collection.update(
                ids=[article_id],
                documents=[document],
                metadatas=[prepared_metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating article in ChromaDB: {e}")
            return False
    
    def delete_article(self, collection_name: str, article_id: str) -> bool:
        """Delete an article from a ChromaDB collection"""
        collection = self.get_collection(collection_name)
        if not collection:
            return False
        
        try:
            collection.delete(ids=[article_id])
            return True
        except Exception as e:
            print(f"Error deleting article from ChromaDB: {e}")
            return False
    
    def get_articles(self, collection_name: str, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None, limit: Optional[int] = None, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Get articles from a ChromaDB collection with optional filtering and embedding inclusion."""
        collection = self.get_collection(collection_name)
        if not collection:
            return []
        
        include_fields = ["metadatas", "documents"]
        if include_embeddings:
            include_fields.append("embeddings")

        try:
            # Build the parameters for the get call
            get_params = {
                "include": include_fields
            }
            
            # Only add parameters if they are not None
            if ids is not None:
                get_params["ids"] = ids
            if where is not None:
                get_params["where"] = where
            if limit is not None:
                get_params["limit"] = limit
                
            results = collection.get(**get_params)
            
        except Exception as e:
            if include_embeddings:
                # Try without embeddings as a fallback
                print(f"Failed to get embeddings, trying without embeddings: {e}")
                try:
                    include_fields = ["metadatas", "documents"]
                    get_params["include"] = include_fields
                    results = collection.get(**get_params)
                    include_embeddings = False  # Mark that we don't have embeddings
                except Exception as e2:
                    print(f"Error retrieving articles from ChromaDB: {e2}")
                    return []
            else:
                print(f"Error retrieving articles from ChromaDB: {e}")
                return []
        
        # Format results
        articles = []
        if results and results.get("ids"):
            num_results = len(results["ids"])
            for i in range(num_results):
                article_id = results["ids"][i]
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                metadata = self._deserialize_metadata(metadata)
                document = results["documents"][i] if results.get("documents") else ""
                
                article_data = {
                    "id": article_id,
                    "document": document,
                    "metadata": metadata
                }

                if include_embeddings and results.get("embeddings") is not None and i < len(results["embeddings"]) and results["embeddings"][i] is not None:
                    article_data["embedding"] = results["embeddings"][i]
                
                articles.append(article_data)
        return articles
    
    def search_articles(self, collection_name: str, query: str, where: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for articles in a ChromaDB collection by semantic similarity"""
        collection = self.get_collection(collection_name)
        if not collection:
            return []
        
        try:
            results = collection.query(
                query_texts=[query],
                where=where,
                n_results=limit
            )
            # Format results
            articles = []
            if results and "ids" in results:
                for i, result_ids in enumerate(results["ids"]):
                    for j, article_id in enumerate(result_ids):
                        metadata = results["metadatas"][i][j] if "metadatas" in results else {}
                        # Deserialize any JSON strings in metadata
                        metadata = self._deserialize_metadata(metadata)
                        document = results["documents"][i][j] if "documents" in results else ""
                        distance = results["distances"][i][j] if "distances" in results else None
                        articles.append({
                            "id": article_id,
                            "document": document,
                            "metadata": metadata,
                            "distance": distance
                        })
            return articles
        except Exception as e:
            print(f"Error searching articles in ChromaDB: {e}")
            return [] 