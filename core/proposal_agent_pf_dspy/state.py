import uuid
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from queue import Queue
from pydantic import BaseModel, Field

# --- Pydantic models for structured data (unchanged) ---
class KnowledgeGap(BaseModel):
    synthesized_summary: str = Field(description="A coherent summary of all reviewed literature.")
    knowledge_gap: str = Field(description="A specific, actionable gap in the current body of knowledge.")
    is_novel: bool = Field(description="An assessment of whether the identified gap is truly novel.")

class Critique(BaseModel):
    score: float = Field(description="A score from 0.0 to 1.0 for the review aspect.")
    justification: str = Field(description="A clear justification for the given score.")

# --- PocketFlow-compatible typed state management ---

@dataclass
class ProposalWorkflowState:
    """Typed state management compatible with PocketFlow's shared parameter"""
    topic: str
    collection_name: str
    
    # Core workflow state
    search_queries: List[str] = field(default_factory=list)
    literature_summaries: List[str] = field(default_factory=list)
    knowledge_gap: Optional[KnowledgeGap] = None
    proposal_draft: str = ""
    review_team_feedback: Dict[str, Critique] = field(default_factory=dict)
    is_approved: bool = False
    revision_cycles: int = 0
    
    # Workflow control
    next_step_index: int = 0
    last_interrupt_type: Optional[str] = None
    
    # PocketFlow HITL communication
    chat_queue: Optional[Queue] = None
    flow_queue: Optional[Queue] = None
    conversation_id: Optional[str] = None
    _last_user_input: Optional[str] = None
    
    # Legacy compatibility
    thread_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields and defaults"""
        if self.conversation_id is None:
            self.conversation_id = str(uuid.uuid4())
        if self.thread_id is None:
            self.thread_id = self.conversation_id
    
    def to_shared_dict(self) -> dict:
        """Convert to PocketFlow's shared dictionary format"""
        data = {}
        
        # Manually copy fields to avoid pickle issues with Queue objects
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            
            # Handle non-serializable objects (like Queue)
            if isinstance(value, Queue):
                # Skip Queue objects in serialization - they'll be managed by orchestrator
                continue
            elif isinstance(value, KnowledgeGap) and value is not None:
                data[field_name] = value.model_dump()
            elif isinstance(value, dict) and field_name == 'review_team_feedback':
                # Handle dictionary of Critique objects
                serialized_feedback = {}
                for k, v in value.items():
                    if isinstance(v, Critique):
                        serialized_feedback[k] = v.model_dump()
                    else:
                        serialized_feedback[k] = v
                data[field_name] = serialized_feedback
            else:
                data[field_name] = value
        
        return data
    
    @classmethod
    def from_shared_dict(cls, shared: dict) -> 'ProposalWorkflowState':
        """Create typed state from PocketFlow's shared dictionary"""
        # Filter out fields that don't belong to our state class
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_shared = {k: v for k, v in shared.items() if k in valid_fields}
        
        # Handle Pydantic model deserialization
        if 'knowledge_gap' in filtered_shared and isinstance(filtered_shared['knowledge_gap'], dict):
            filtered_shared['knowledge_gap'] = KnowledgeGap(**filtered_shared['knowledge_gap'])
        
        if 'review_team_feedback' in filtered_shared and isinstance(filtered_shared['review_team_feedback'], dict):
            feedback = {}
            for k, v in filtered_shared['review_team_feedback'].items():
                if isinstance(v, dict):
                    feedback[k] = Critique(**v)
                else:
                    feedback[k] = v
            filtered_shared['review_team_feedback'] = feedback
        
        return cls(**filtered_shared)
    
    def update(self, **kwargs) -> None:
        """Type-safe state updates with validation"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Add type checking here if needed
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid state field: {key}")
    
    def append_to(self, key: str, value: Any) -> None:
        """Appends a value to a list-based state variable"""
        if not hasattr(self, key):
            raise ValueError(f"Invalid state field: {key}")
        
        current_value = getattr(self, key)
        if isinstance(current_value, list):
            current_value.append(value)
        else:
            raise TypeError(f"'{key}' is not a list and cannot be appended to.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Legacy compatibility method - converts to dictionary"""
        return self.to_shared_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProposalWorkflowState':
        """Legacy compatibility method - creates from dictionary"""
        return cls.from_shared_dict(data)

def create_shared_state(topic: str, collection_name: str, 
                       chat_queue: Optional[Queue] = None, 
                       flow_queue: Optional[Queue] = None) -> dict:
    """Create initial shared state dictionary for PocketFlow"""
    state = ProposalWorkflowState(
        topic=topic,
        collection_name=collection_name,
        chat_queue=chat_queue or Queue(),
        flow_queue=flow_queue or Queue()
    )
    return state.to_shared_dict()

# --- Legacy compatibility alias ---
WorkflowState = ProposalWorkflowState 