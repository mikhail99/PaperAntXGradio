from datetime import datetime
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from core.pocketflow_demo.types import SharedState, SessionState, ResearchContext, JourneyEntry
from pocketflow import Node
from queue import Queue
from typing import Optional, Tuple

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
        """Helper to log messages to flow queue"""
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(message)

class ReviewNodeBase(NodeBase):
    """Special base class for human-in-the-loop review nodes"""
    
    def prep(self, shared: SharedState) -> Tuple[str, Queue[str]]:
        # Signal end of automated flow
        flow_log: Queue[Optional[str]] = shared["flow_queue"]
        flow_log.put(None)  # Stop flow thoughts
        
        # Get the result to review
        context = self.get_research_context(shared)
        review_content = self.get_review_content(context)
        
        review_message = self.format_review_message(review_content)
        return review_message, shared["queue"]
    
    def exec(self, prep_res: Tuple[str, Queue[str]]) -> str:
        # Send message to user interface
        review_message, queue = prep_res
        queue.put(review_message)
        # Don't put None in chat queue - that's handled by the UI layer
        return review_message
    
    def post(self, shared: SharedState, prep_res: Tuple[str, Queue[str]], exec_res: str) -> str:
        # Review nodes DON'T update research journey
        # They only update process state
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        session["waiting_for_feedback"] = self.get_review_type()
        save_conversation(conversation_id, session)
        
        return "done"  # Always pause execution
    
    def get_review_content(self, context: ResearchContext) -> str:
        """Override in subclass to specify what content to review"""
        raise NotImplementedError("Subclass must implement get_review_content")
    
    def format_review_message(self, content: str) -> str:
        """Override in subclass to format the review message"""
        raise NotImplementedError("Subclass must implement format_review_message")
    
    def get_review_type(self) -> str:
        """Override in subclass to specify the review type for feedback tracking"""
        raise NotImplementedError("Subclass must implement get_review_type")

def check_feedback_in_message(message: str) -> Optional[str]:
    """
    Check if a message contains feedback keywords.
    Returns: 'approved', 'rejected', or None
    """
    if not message:
        return None
        
    message_lower = message.lower().strip()
    print(f"🔍 Checking message for feedback: '{message_lower}'")
    
    # Check for approval keywords (more flexible)
    approval_keywords = ["approve", "approved", "proceed", "continue", "yes", "ok", "okay", "good", "looks good"]
    rejection_keywords = ["reject", "rejected", "retry", "redo", "no", "not good", "bad", "change"]
    
    for keyword in approval_keywords:
        if keyword in message_lower:
            print(f"✅ Detected APPROVAL keyword: '{keyword}'")
            return "approved"
    
    for keyword in rejection_keywords:
        if keyword in message_lower:
            print(f"❌ Detected REJECTION keyword: '{keyword}'")
            return "rejected"
    
    print("❓ No feedback keywords detected")
    return None 