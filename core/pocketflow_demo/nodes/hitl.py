import threading
from core.pocketflow_demo.nodes.actions import Action
from core.pocketflow_demo.nodes.base import NodeBase
from core.pocketflow_demo.types import SharedState, ResearchContext
from typing import Tuple, Any, Optional
from queue import Queue

class ReviewQueries(NodeBase):
    """Review node for search queries - presents results for human approval"""
    
    @staticmethod
    def required_params():
        return []
    
    @staticmethod
    def action_type():
        return Action.review_queries
    
    def prep(self, shared: SharedState) -> Tuple[str, Queue[Optional[str]]]:
        # Get the queries to review
        context = self.get_research_context(shared)
        queries = context.get("queries") or "No queries generated"
        
        # Format the review message (keep current UI style)
        review_message = f"""🔍 **Generated Search Queries:**

{queries}

**Please review these queries:**
👍 Say **"approved"** or **"ok"** to proceed with literature review
👎 Say **"rejected"** or **"redo"** to generate different queries"""
        
        return review_message, shared["queue"]
    
    def exec(self, prep_res: Tuple[str, Queue[Optional[str]]]) -> str:
        review_message, queue = prep_res
        
        # Send message to UI
        queue.put(review_message)
        queue.put(None)  # type: ignore  # Queue[str] but we need None for UI termination
        
        return review_message
    
    def post(self, shared: SharedState, prep_res: Tuple[str, Queue[Optional[str]]], exec_res: str) -> str:
        # Stop flow thoughts
        self.log_to_flow(shared, None)
        
        # Simple threading-based HITL (like their approach)
        shared_dict = shared  # type: ignore  # Access as dict for dynamic fields
        if "hitl_event" not in shared_dict:
            shared_dict["hitl_event"] = threading.Event()  # type: ignore  # Dynamic field
        
        hitl_event = shared_dict["hitl_event"]  # type: ignore  # Dynamic field access
        hitl_event.clear()  # type: ignore  # Reset event
        
        print("ReviewQueries: Waiting for user input...")
        hitl_event.wait()  # BLOCK here until user provides feedback
        print(f"ReviewQueries: Received input: {shared_dict.get('user_input', 'N/A')}")
        
        # Process feedback
        user_input = str(shared_dict.get("user_input", "")).lower().strip()
        if any(word in user_input for word in ["approved", "ok", "proceed", "continue", "yes"]):
            print("✅ Queries approved - proceeding to literature review")
            return "approved"
        elif any(word in user_input for word in ["rejected", "redo", "retry", "no"]):
            print("❌ Queries rejected - regenerating")
            return "rejected"
        else:
            print("❓ Unclear feedback - asking for clarification")
            return "default"

class ReviewReport(NodeBase):
    """Review node for final report - presents results for human approval"""
    
    @staticmethod
    def required_params():
        return []
    
    @staticmethod
    def action_type():
        return Action.review_report
    
    def prep(self, shared: SharedState) -> Tuple[str, Queue[Optional[str]]]:
        # Get the proposal to review
        context = self.get_research_context(shared)
        proposal = context.get("proposal") or "No proposal generated"
        
        # Format the review message (keep current UI style)
        review_message = f"""📋 **Final Research Proposal:**

{proposal}

**Please review the proposal:**
👍 Say **"approved"** or **"ok"** to finalize and complete
👎 Say **"rejected"** or **"revise"** to improve the proposal"""
        
        return review_message, shared["queue"]
    
    def exec(self, prep_res: Tuple[str, Queue[Optional[str]]]) -> str:
        review_message, queue = prep_res
        
        # Send message to UI
        queue.put(review_message)
        queue.put(None)  # type: ignore  # Queue[str] but we need None for UI termination
        
        return review_message
    
    def post(self, shared: SharedState, prep_res: Tuple[str, Queue[Optional[str]]], exec_res: str) -> str:
        # Stop flow thoughts
        self.log_to_flow(shared, None)
        
        # Simple threading-based HITL (like their approach)
        shared_dict = shared  # type: ignore  # Access as dict for dynamic fields
        if "hitl_event" not in shared_dict:
            shared_dict["hitl_event"] = threading.Event()  # type: ignore  # Dynamic field
        
        hitl_event = shared_dict["hitl_event"]  # type: ignore  # Dynamic field access
        hitl_event.clear()  # type: ignore  # Reset event
        
        print("ReviewReport: Waiting for user input...")
        hitl_event.wait()  # BLOCK here until user provides feedback
        print(f"ReviewReport: Received input: {shared_dict.get('user_input', 'N/A')}")
        
        # Process feedback
        user_input = str(shared_dict.get("user_input", "")).lower().strip()
        if any(word in user_input for word in ["approved", "ok", "proceed", "continue", "yes"]):
            print("✅ Report approved - finalizing")
            return "approved"
        elif any(word in user_input for word in ["rejected", "revise", "retry", "no"]):
            print("❌ Report rejected - regenerating")
            return "rejected"
        else:
            print("❓ Unclear feedback - asking for clarification")
            return "default"
