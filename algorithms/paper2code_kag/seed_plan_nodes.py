import logging
from typing import Dict, List, Any, TypedDict

from pocketflow import Node
from utils.knowledge_base import KnowledgeBase
from utils.knowledge_retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)

# --- TypedDict Definitions for Shared State ---

class DocumentInfo(TypedDict):
    doc_name: str
    full_text: str
    abstract: str # Assuming we can get this

class SimilarPaper(TypedDict):
    doc_name: str
    citation: str
    score: float

class RetrievedAbstraction(TypedDict):
    retrieved_abstraction: Dict[str, Any] # The original abstraction object
    relevance_score: float
    source_count: int
    source_documents: List[str]

class SeedPlanSharedState(TypedDict, total=False):
    """The shared state for the plan seeding flow."""
    # Input
    new_document: DocumentInfo
    
    # Output of FindSimilarPapersNode
    similar_papers: List[SimilarPaper]
    
    # Output of RetrieveKnownAbstractionsNode
    retrieved_abstractions: List[RetrievedAbstraction]
    
    # Output of AssembleDraftPlanNode
    draft_plan: Dict[str, Any]
    draft_plan_saved: bool
    draft_plan_path: str

# --- Node Implementations ---

class FindSimilarPapersNode(Node):
    """
    Finds papers in the Knowledge Base similar to the new document's abstract.
    """
    def __init__(self, top_n: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.top_n = top_n

    def prep(self, shared: SeedPlanSharedState) -> str:
        """Prepare the abstract of the new document for querying."""
        if "new_document" not in shared or "abstract" not in shared["new_document"]:
            raise ValueError("Missing 'new_document' or its 'abstract' in shared state.")
        logger.info(f"Preparing to find papers similar to '{shared['new_document']['doc_name']}'.")
        return shared["new_document"]["abstract"]

    def exec(self, query_text: str) -> List[SimilarPaper]:
        """Execute the search for similar papers."""
        kb = KnowledgeBase()
        retriever = KnowledgeRetriever(kb)
        similar_papers = retriever.find_similar_papers(query_text, top_n=self.top_n)
        return similar_papers

    def post(self, shared: SeedPlanSharedState, prep_res: str, exec_res: List[SimilarPaper]):
        """Save the list of similar papers to the shared state."""
        shared["similar_papers"] = exec_res
        logger.info(f"Found {len(exec_res)} similar papers.")
        return "default"

class RetrieveKnownAbstractionsNode(Node):
    """
    Retrieves and scores known abstractions relevant to the new document's content.
    """
    def __init__(self, top_n_chunks: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.top_n_chunks = top_n_chunks

    def prep(self, shared: SeedPlanSharedState) -> str:
        """Prepare the full text of the new document for retrieval."""
        if "new_document" not in shared or "full_text" not in shared["new_document"]:
            raise ValueError("Missing 'new_document' or its 'full_text' in shared state.")
        logger.info(f"Preparing to retrieve abstractions for '{shared['new_document']['doc_name']}'.")
        # For simplicity, we use the full text. In a real scenario, this would be chunked.
        return shared["new_document"]["full_text"]

    def exec(self, full_text: str) -> List[RetrievedAbstraction]:
        """Execute the retrieval and scoring of abstractions."""
        kb = KnowledgeBase()
        retriever = KnowledgeRetriever(kb)
        retrieved_abstractions = retriever.retrieve_and_score_abstractions(
            text_chunk=full_text, 
            top_n_chunks=self.top_n_chunks
        )
        return retrieved_abstractions

    def post(self, shared: SeedPlanSharedState, prep_res: str, exec_res: List[RetrievedAbstraction]):
        """Save the retrieved abstractions to the shared state."""
        shared["retrieved_abstractions"] = exec_res
        logger.info(f"Retrieved {len(exec_res)} scored and deduplicated abstractions.")
        return "default"

class AssembleDraftPlanNode(Node):
    """
    Assembles the retrieved information into a structured draft plan.
    """
    def prep(self, shared: SeedPlanSharedState) -> Dict[str, Any]:
        """Prepare the retrieved data for assembly."""
        if "similar_papers" not in shared or "retrieved_abstractions" not in shared:
            raise ValueError("Missing 'similar_papers' or 'retrieved_abstractions' in shared state.")
        
        logger.info("Preparing to assemble draft plan.")
        return {
            "source_doc": shared.get("new_document", {}).get("doc_name", "Unknown"),
            "similar_papers": shared["similar_papers"],
            "retrieved_abstractions": shared["retrieved_abstractions"]
        }

    def exec(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the draft plan dictionary."""
        draft_plan = {
            "metadata": {
                "source_document": plan_data["source_doc"],
                "plan_type": "seeded_draft",
                "status": "success"
            },
            "seeding_results": {
                "similar_papers_found": len(plan_data["similar_papers"]),
                "retrieved_abstractions_found": len(plan_data["retrieved_abstractions"]),
                "similar_papers": plan_data["similar_papers"],
                "retrieved_abstractions": plan_data["retrieved_abstractions"]
            }
        }
        logger.info("Successfully assembled draft plan.")
        return draft_plan

    def post(self, shared: SeedPlanSharedState, prep_res: Dict[str, Any], exec_res: Dict[str, Any]):
        """Save the draft plan to the shared state."""
        shared["draft_plan"] = exec_res
        logger.info("Saved draft plan to shared state.")
        # This node could also save the plan to a file if needed.
        return "default" 