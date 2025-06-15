import logging
from collections import defaultdict
from typing import List, Dict, Any

from paperqa import Docs
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class KnowledgeRetriever:
    """
    A utility class to retrieve and rank information from the Knowledge Base.
    It encapsulates the logic for querying PaperQA and the abstractions database.
    """

    def __init__(self, kb: KnowledgeBase):
        """
        Initializes the retriever with a KnowledgeBase instance.
        
        Args:
            kb: An initialized KnowledgeBase object.
        """
        self.kb = kb
        self.docs: Docs = self.kb.load_docs()
        self.abstractions_db: Dict[str, Any] = self.kb.load_abstractions()

    def find_similar_papers(self, query_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Finds the most similar papers in the PaperQA index based on a query text.
        
        Args:
            query_text: The text to use for the semantic search (e.g., an abstract).
            top_n: The number of similar papers to return.
            
        Returns:
            A list of dictionaries, each containing information about a similar paper.
        """
        logger.info(f"Querying for {top_n} papers similar to query: '{query_text[:100]}...'")
        
        # PaperQA's query method returns an Answer object. We need to parse its 'docs' attribute.
        try:
            # We don't need a real answer, just the context documents.
            answer = self.docs.query(query_text, k=top_n, max_sources=top_n)
            
            similar_docs = []
            if not hasattr(answer, 'docs') or not answer.docs:
                logger.warning("PaperQA query did not return any context documents.")
                return []

            for doc_name, doc_obj in answer.docs.items():
                similar_docs.append({
                    "doc_name": doc_name,
                    "citation": doc_obj.citation,
                    "score": "N/A" # PaperQA's Answer object doesn't expose a simple similarity score per doc.
                })
            
            logger.info(f"Found {len(similar_docs)} similar papers.")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Failed to query PaperQA for similar papers: {e}", exc_info=True)
            return []

    def retrieve_and_score_abstractions(self, text_chunk: str, top_n_chunks: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves, aggregates, and scores abstractions related to a given text chunk.
        
        1. Finds similar text chunks from the entire PaperQA corpus.
        2. Retrieves all abstractions linked to the source documents of those chunks.
        3. Scores and deduplicates the abstractions based on relevance and frequency.
        
        Args:
            text_chunk: The text to find relevant abstractions for.
            top_n_chunks: The number of similar text chunks to consider from the corpus.
            
        Returns:
            A sorted list of deduplicated and scored abstractions.
        """
        logger.info(f"Retrieving abstractions for text chunk: '{text_chunk[:100]}...'")
        
        try:
            answer = self.docs.query(text_chunk, k=top_n_chunks, max_sources=top_n_chunks)
            
            if not hasattr(answer, 'contexts') or not answer.contexts:
                logger.warning("PaperQA query did not return any contexts for abstraction retrieval.")
                return []

            # Aggregate abstractions from all relevant source documents
            scored_abstractions = defaultdict(lambda: {"count": 0, "score": 0.0, "sources": set()})
            
            for context in answer.contexts:
                source_doc_name = context.doc.docname
                # The context object gives a score!
                relevance_score = context.score 
                
                # Get all abstractions for this source document
                doc_abstractions = self.abstractions_db.get("abstractions_by_doc", {}).get(source_doc_name, [])

                for abstraction in doc_abstractions:
                    # Use abstraction name as a unique key for deduplication
                    abs_name = abstraction.get("abstraction", {}).get("name", "").lower()
                    if not abs_name:
                        continue
                        
                    # Aggregate scores and count occurrences
                    scored_abstractions[abs_name]["count"] += 1
                    scored_abstractions[abs_name]["score"] += relevance_score
                    scored_abstractions[abs_name]["sources"].add(source_doc_name)
                    # Store the original full abstraction object, preferring the one from the most relevant source
                    if scored_abstractions[abs_name].get("best_score", 0.0) < relevance_score:
                         scored_abstractions[abs_name]["object"] = abstraction
                         scored_abstractions[abs_name]["best_score"] = relevance_score

            # Normalize and format the results
            final_list = []
            for name, data in scored_abstractions.items():
                if "object" not in data: continue
                
                final_list.append({
                    "retrieved_abstraction": data["object"],
                    "relevance_score": data["score"] / data["count"], # Average score
                    "source_count": data["count"],
                    "source_documents": list(data["sources"])
                })

            # Sort by a combination of relevance and frequency
            final_list.sort(key=lambda x: x["relevance_score"] * x["source_count"], reverse=True)
            
            logger.info(f"Retrieved and scored {len(final_list)} unique abstractions.")
            return final_list

        except Exception as e:
            logger.error(f"Failed to retrieve and score abstractions: {e}", exc_info=True)
            return [] 