import uuid
from typing import List, Dict, Any, TypeVar
from pydantic import BaseModel, Field

# --- Pydantic models for structured data ---
class KnowledgeGap(BaseModel):
    synthesized_summary: str = Field(description="A coherent summary of all reviewed literature.")
    knowledge_gap: str = Field(description="A specific, actionable gap in the current body of knowledge.")
    is_novel: bool = Field(description="An assessment of whether the identified gap is truly novel.")

class Critique(BaseModel):
    score: float = Field(description="A score from 0.0 to 1.0 for the review aspect.")
    justification: str = Field(description="A clear justification for the given score.")

# --- The main state manager ---
T = TypeVar('T')

class WorkflowState:
    """A typed, class-based state manager for the proposal workflow."""
    
    def __init__(self, topic: str, collection_name: str):
        self.thread_id: str = str(uuid.uuid4())
        self.topic: str = topic
        self.collection_name: str = collection_name
        
        # --- Core state variables ---
        self.search_queries: List[str] = []
        self.literature_summaries: List[str] = []
        self.knowledge_gap: KnowledgeGap | None = None
        self.proposal_draft: str = ""
        self.review_team_feedback: Dict[str, Critique] = {}
        self.is_approved: bool = False
        self.revision_cycles: int = 0
        
        # --- Workflow control ---
        self.next_step_index: int = 0
        self.last_interrupt_type: str | None = None
        
    def update(self, key: str, value: Any) -> None:
        """Safely updates a state variable."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' is not a valid state variable.")

    def append_to(self, key: str, value: T) -> None:
        """Appends a value to a list-based state variable."""
        current_value = getattr(self, key, None)
        if isinstance(current_value, list):
            current_value.append(value)
        else:
            raise TypeError(f"'{key}' is not a list and cannot be appended to.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the state to a dictionary for persistence."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Deserializes the state from a dictionary."""
        instance = cls(data['topic'], data['collection_name'])
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    def __repr__(self) -> str:
        return f"<WorkflowState queries={len(self.search_queries)} summaries={len(self.literature_summaries)}>" 