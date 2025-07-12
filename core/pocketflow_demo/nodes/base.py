from datetime import datetime
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from core.pocketflow_demo.types import SharedState, SessionState, ResearchContext, JourneyEntry
from pocketflow import Node
from queue import Queue
from typing import Optional

class NodeBase(Node):
    """Base class for all research workflow nodes with helper methods"""
    
    def get_research_context(self, shared: SharedState) -> ResearchContext:
        """Extract research context from journey queue"""
        session = load_conversation(shared["conversation_id"])
        journey = session.get("research_journey", [])
        
        return ResearchContext(
            topic=self._find_step_result(journey, "topic_extraction"),
            queries=self._find_step_result(journey, "query_generation"),
            literature=self._find_step_result(journey, "literature_review"),
            gaps=self._find_step_result(journey, "gap_analysis"),
            proposal=self._find_step_result(journey, "proposal_generation"),
            all_steps=journey,  # Full history if needed
            original_query=self._get_original_query(shared, journey)
        )
    
    def _find_step_result(self, journey: list[JourneyEntry], step_name: str) -> Optional[str]:
        """Find the result of a specific step in the research journey"""
        for entry in reversed(journey):  # Start from most recent
            if entry.get("step") == step_name:
                return entry.get("result")
        return None
    
    def _get_original_query(self, shared: SharedState, journey: list[JourneyEntry]) -> str:
        """Get the original user query that started this research"""
        # First try journey
        topic = self._find_step_result(journey, "topic_extraction")
        if topic:
            return topic
        
        # Fallback to current query
        return shared.get("query", "")
    
    def update_research_journey(self, shared: SharedState, step_name: str, result: str) -> None:
        """Standard way to update research journey"""
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        if "research_journey" not in session:
            session["research_journey"] = []
            
        journey_entry = JourneyEntry(
            step=step_name,
            result=result,
            timestamp=datetime.now().isoformat(),
            action=str(self.action_type()) if self.action_type() else "unknown"
        )
        
        session["research_journey"].append(journey_entry)
        save_conversation(conversation_id, session)
    
    def log_to_flow(self, shared: SharedState, message: Optional[str]) -> None:
        """Helper to log messages to flow queue (accepts None to signal completion)"""
        flow_log: Queue[Optional[str]] = shared["flow_queue"]
        flow_log.put(message) 