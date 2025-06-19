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

# --- The Global State ---

class ProposalAgentState(TypedDict, total=False):

    # Collection
    collection_id: str
    # Core Research
    topic: str
    search_queries: Annotated[list[str], operator.add]

    literature_summaries: Annotated[list[str], operator.add]
    reflection: SummaryReflection

    # Planning & Design
    research_plan:  Annotated[list[str], operator.add]
    novelty_assessment: Annotated[list[NoveltyAssessment], operator.add] 


    # writing proposals and experiments
    #experiment_protocol:  ExperimentProtocol
    #markdown_proposal: str
    
    
