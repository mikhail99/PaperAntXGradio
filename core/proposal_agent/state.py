import operator
from typing import TypedDict, List
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

class QueryList(BaseModel):
    """A list of search queries."""
    queries: List[str] = Field(description="A list of search queries.")

class Reflection(BaseModel):
    """Reflection on the gathered information."""
    is_sufficient: bool = Field(description="Is the literature review sufficient?")
    knowledge_gap: str = Field(description="Identified gaps in the literature.")
    follow_up_queries: List[str] = Field(description="New queries to address gaps.")

class NoveltyAssessment(BaseModel):
    """Assessment of the research plan's novelty."""
    is_novel: bool = Field(description="Is the research plan novel enough to proceed?")
    similar_papers: list[str] = Field(description="List of URLs for similar papers found.")
    justification: str = Field(description="Justification for the novelty assessment.")

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


# --- The Global State ---

class ProposalAgentState(TypedDict, total=False):
    """The complete state for the proposal generation agent."""

    # Core Research
    topic: str
    search_queries: Annotated[list[str], operator.add]
    papers: Annotated[list[Paper], operator.add]
    literature_summary: str
    reflection: Reflection

    # Planning & Design
    research_plan: str
    novelty_assessment: NoveltyAssessment
    experiment_protocol: ExperimentProtocol

    # Writing & Refinement
    latex_proposal: str
    critique: ProposalCritique
    
    # Control Flow
    messages: list
    max_loops: int
    loop_count: int
    local_papers_only: bool
    
    # Paper Limits
    research_papers_count: int
    reflection_papers_count: int
