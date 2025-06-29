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
    """A list of search queries."""
    queries: List[str]

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
    The overall state of the proposal generation agent.
    It's passed between nodes in the graph.
    
    Based on LangGraph documentation, reducers are only needed when you want
    to customize how updates are applied. For most fields, the default behavior
    (overwrite) is sufficient. Only lists need special handling.
    """
    # --- Inputs ---
    topic: str
    collection_name: str
    local_papers_only: bool

    # --- Agent Trajectory ---
    
    # 1. Query Generation
    search_queries: Annotated[List[str], operator.add]

    # 2. Literature Review
    # Raw summaries from each literature review agent run
    literature_summaries: Annotated[List[str], operator.add]
    # The single, synthesized summary and knowledge gap from the aggregator
    knowledge_gap: KnowledgeGap
    
    # 3. Proposal Formulation
    proposal_draft: str
    
    # 4. Proposal Review
    # Raw feedback from each review team member, keyed by member name (e.g., "review_feasibility")
    review_team_feedback: Annotated[Dict[str, Any], merge_feedback]
    # The final, synthesized review from the aggregator
    final_review: FinalReview
    
    # --- Human in the Loop ---
    human_feedback: Optional[str]
    paused_on: Optional[str] # The name of the node we paused on

    # --- Loop Control ---
    # We can add counters here if we need to enforce max loops for revisions
    proposal_revision_cycles: int

    current_literature: str

    
