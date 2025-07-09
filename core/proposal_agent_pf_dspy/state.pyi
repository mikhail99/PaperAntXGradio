import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type, TypeVar, ClassVar
from queue import Queue
from pydantic import BaseModel, Field

# Type annotations for Pydantic models
class KnowledgeGap(BaseModel):
    synthesized_summary: str
    knowledge_gap: str
    is_novel: bool

class Critique(BaseModel):
    score: float
    justification: str

# Type variable for state class
T = TypeVar('T', bound='ProposalWorkflowState')

@dataclass
class ProposalWorkflowState:
    """Typed state management compatible with PocketFlow's shared parameter"""
    topic: str
    collection_name: str
    
    # Core workflow state
    search_queries: List[str]
    literature_summaries: List[str]
    knowledge_gap: Optional[KnowledgeGap]
    proposal_draft: str
    review_team_feedback: Dict[str, Critique]
    is_approved: bool
    revision_cycles: int
    
    # Workflow control
    next_step_index: int
    last_interrupt_type: Optional[str]
    
    # PocketFlow HITL communication
    chat_queue: Optional[Queue[str]]
    flow_queue: Optional[Queue[str]]
    conversation_id: Optional[str]
    _last_user_input: Optional[str]
    
    # Legacy compatibility
    thread_id: Optional[str]
    
    # Dataclass field definitions
    __dataclass_fields__: ClassVar[Dict[str, Any]]
    
    def __post_init__(self) -> None: ...
    
    def to_shared_dict(self) -> Dict[str, Any]: ...
    
    @classmethod
    def from_shared_dict(cls: Type[T], shared: Dict[str, Any]) -> T: ...
    
    def update(self, **kwargs: Any) -> None: ...
    
    def append_to(self, key: str, value: Any) -> None: ...
    
    def to_dict(self) -> Dict[str, Any]: ...
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T: ...

def create_shared_state(
    topic: str, 
    collection_name: str, 
    chat_queue: Optional[Queue[str]] = None, 
    flow_queue: Optional[Queue[str]] = None
) -> Dict[str, Any]: ...

# Legacy compatibility alias
WorkflowState = ProposalWorkflowState 