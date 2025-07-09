import operator
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field


# --- Pydantic Models for State ---

class Paper(BaseModel):
    """Represents a single research paper."""
    id: str
    title: str
    summary: str
    authors: list[str]
    url: str

class SingleQuery(BaseModel):
    """A single search query."""
    query: str = Field(description="A single search query.")

class SummaryReflection(BaseModel):
    """Reflection on the gathered information."""
    is_sufficient: bool = Field(description="Is the literature review sufficient?")
    knowledge_gap: str = Field(description="Identified gaps in the literature.")
    follow_up_queries: List[str] = Field(description="New queries to address gaps.")

class NoveltyAssessment(BaseModel):
    """Assessment of the research plan's novelty."""
    is_novel: bool = Field(description="Is the research plan novel enough to proceed?")
    justification: str = Field(description="Justification for the novelty assessment.")
    similar_papers: list[dict] = Field(default=[], description="List of similar papers found, with title and url.")
'''
class ExperimentProtocol(BaseModel):
    """Detailed plan for experiments."""
    methodology: str = Field(description="The proposed methodology.")
    datasets: list[str] = Field(description="Datasets to be used.")
    evaluation_metrics: list[str] = Field(description="Metrics for evaluation.")
    pseudocode: str = Field(description="Pseudocode for the core algorithm.")

class ProposalCritique(BaseModel):
    """Structured feedback from the review phase."""
    strengths: str = Field(description="Positive aspects of the proposal.")
    weaknesses: str = Field(description="Areas for improvement.")
    revision_required: bool = Field(description="Is a revision necessary?")
    suggested_changes: str = Field(description="Specific suggestions for revision.")
'''

# --- Data-only Schemas for LLM outputs ---
# These are used for type hinting and structured output parsing

class QueryList(TypedDict):
    """Output from query generation - contains a single focused query."""
    queries: List[str]  # Will contain one query, but keep as list for compatibility

class KnowledgeGap(TypedDict):
    """The output of the literature synthesis step."""
    synthesized_summary: str
    knowledge_gap: str
    justification: str
    is_novel: bool
    similar_papers: Optional[List[Dict]]

class Critique(TypedDict):
    """The structured feedback from a single proposal reviewer."""
    score: float 
    justification: str

class FinalReview(TypedDict):
    """The final, synthesized review from the proposal review aggregator."""
    is_approved: bool
    critique: str

def merge_feedback(
    left: Dict[str, Any] | None = None, right: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Merges feedback dictionaries from parallel review nodes."""
    if left is None:
        left = {}
    if right is None:
        right = {}
    result = left.copy()
    result.update(right)
    return result

# --- Main Agent State ---

class ProposalAgentState(TypedDict):
    """
    The state definition for our multi-agent research proposal generation system.
    This state is passed between all nodes in the graph.
    """
    # Input configuration
    topic: str
    collection_name: str
    local_papers_only: bool
    
    # Query generation state
    search_queries: List[str]  # Single query replacement, not accumulation
    
    # Literature review state  
    literature_summaries: Annotated[List[str], operator.add]
    current_literature: str
    
    # Knowledge synthesis state
    knowledge_gap: Dict[str, Any]
    
    # Proposal creation state
    proposal_draft: str
    
    # Review state
    review_team_feedback: Annotated[Dict[str, Any], merge_feedback]
    final_review: Dict[str, Any]
    
    # Revision tracking
    proposal_revision_cycles: int

    
