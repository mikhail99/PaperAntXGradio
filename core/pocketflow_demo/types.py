from typing import TypedDict, List, Dict, Any, Optional, Union
from typing_extensions import Annotated
from queue import Queue
from datetime import datetime
import threading

# Research Journey Entry Structure
class JourneyEntry(TypedDict):
    step: str
    result: str
    timestamp: str
    action: str

# Custom reducer for appending to research journey
def append_journey(current: List[JourneyEntry], new: JourneyEntry) -> List[JourneyEntry]:
    """Reducer that appends new entries to the research journey"""
    if current is None:
        return [new]
    return current + [new]

# Shared State Type Definition
class SharedState(TypedDict):
    """Typed shared state passed to all PocketFlow nodes"""
    conversation_id: str
    query: str
    history: List[Dict[str, Any]]
    queue: Queue[Optional[str]]  # Allow None for UI termination
    flow_queue: Queue[Optional[str]]  # Flow queue can accept None to signal completion

# Optional HITL fields for SharedState (using dict access, not typed)
# hitl_event: threading.Event
# user_input: str

# Session State Type Definition  
class SessionState(TypedDict, total=False):
    """Typed session state for persistence"""
    last_action: Optional[str]
    waiting_for_feedback: Optional[str]
    research_journey: List[JourneyEntry]

# Research Context Type (extracted from journey)
class ResearchContext(TypedDict, total=False):
    """Structured research context extracted from journey"""
    topic: Optional[str]
    queries: Optional[str]
    literature: Optional[str]
    gaps: Optional[str]
    proposal: Optional[str]
    all_steps: List[JourneyEntry]
    original_query: str 