from core.pocketflow_demo.nodes.actions import Action
from core.pocketflow_demo.nodes.base import NodeBase, check_feedback_in_message
from core.pocketflow_demo.types import SharedState, ResearchContext
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from typing import Union, Tuple
from queue import Queue

class ReviewQueries(NodeBase):
    """Review node for search queries - presents results for human approval"""
    
    @staticmethod
    def required_params():
        return []  # Review nodes don't need input parameters
    
    @staticmethod
    def action_type():
        return Action.review_queries
    
    def prep(self, shared: SharedState) -> Union[Tuple[str, str, Queue[str]], Tuple[str, Queue[str]]]:
        # Check if we're resuming after feedback
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        if session.get("waiting_for_feedback") == "queries":
            # We're resuming - process the feedback
            query = shared.get("query", "")
            feedback = check_feedback_in_message(query)
            print(f"🔍 Processing feedback for queries: '{feedback}'")
            
            if feedback == "approved":
                print("✅ Queries approved - proceeding to literature review")
                # Clear waiting state
                session["waiting_for_feedback"] = None
                save_conversation(conversation_id, session)
                # Return special marker to skip normal review display
                return "FEEDBACK_PROCESSED", "approved", shared["queue"]
            elif feedback == "rejected":
                print("❌ Queries rejected - regenerating")
                session["waiting_for_feedback"] = None
                save_conversation(conversation_id, session)
                return "FEEDBACK_PROCESSED", "rejected", shared["queue"]
            else:
                print("❓ Unclear feedback - asking for clarification")
                session["waiting_for_feedback"] = None
                save_conversation(conversation_id, session)
                return "FEEDBACK_PROCESSED", "default", shared["queue"]
        
        # First time showing review - create review message
        self.log_to_flow(shared, None)  # Stop flow thoughts
        
        # Get the result to review
        context = self.get_research_context(shared)
        review_content = self.get_review_content(context)
        
        review_message = self.format_review_message(review_content)
        return review_message, shared["queue"]
    
    def exec(self, prep_res: Union[Tuple[str, str, Queue[str]], Tuple[str, Queue[str]]]) -> str:
        # Check if we're processing feedback
        if len(prep_res) == 3 and prep_res[0] == "FEEDBACK_PROCESSED":
            _, feedback_result, _ = prep_res
            return feedback_result  # Return the feedback decision
        
        # Normal review display - unpack 2-tuple
        if len(prep_res) == 2:
            review_message, queue = prep_res
            queue.put(review_message)
            queue.put(None)  # Signal UI completion - CRITICAL FOR UI TO NOT HANG!
            return review_message
        
        # Fallback - shouldn't reach here
        return "Error in review processing"
    
    def post(self, shared: SharedState, prep_res: Union[Tuple[str, str, Queue[str]], Tuple[str, Queue[str]]], exec_res: str) -> str:
        # Check if we processed feedback
        if len(prep_res) == 3 and prep_res[0] == "FEEDBACK_PROCESSED":
            return exec_res  # Return the routing decision ("approved", "rejected", "default")
        
        # Normal first-time review - pause execution
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        session["waiting_for_feedback"] = self.get_review_type()
        save_conversation(conversation_id, session)
        
        return "pause"  # Route to PauseForFeedback node instead of "done"
    
    def get_review_content(self, context: ResearchContext) -> str:
        """Get the queries to review from research context"""
        return context.get("queries") or "No queries generated"
    
    def format_review_message(self, content: str) -> str:
        """Format the review message for queries"""
        return f"""🔍 **Generated Search Queries:**

{content}

**Please review these queries:**
👍 Say **"approved"** or **"ok"** to proceed with literature review
👎 Say **"rejected"** or **"redo"** to generate different queries"""
    
    def get_review_type(self) -> str:
        """Return the review type for feedback tracking"""
        return "queries"
    
    def log_to_flow(self, shared: SharedState, message) -> None:
        """Helper to log messages to flow queue"""
        flow_log = shared["flow_queue"]
        flow_log.put(message)
    
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
            all_steps=journey,
            original_query=self._get_original_query(shared, journey)
        )
    
    def _find_step_result(self, journey, step_name):
        """Find the result of a specific step in the research journey"""
        for entry in reversed(journey):
            if entry.get("step") == step_name:
                return entry.get("result")
        return None
    
    def _get_original_query(self, shared, journey):
        """Get the original user query that started this research"""
        topic = self._find_step_result(journey, "topic_extraction")
        if topic:
            return topic
        return shared.get("query", "")

class ReviewReport(NodeBase):
    """Review node for final report - presents results for human approval"""
    
    @staticmethod
    def required_params():
        return []  # Review nodes don't need input parameters
    
    @staticmethod
    def action_type():
        return Action.review_report
    
    def prep(self, shared: SharedState) -> Union[Tuple[str, str, Queue[str]], Tuple[str, Queue[str]]]:
        # Check if we're resuming after feedback
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        if session.get("waiting_for_feedback") == "report":
            # We're resuming - process the feedback
            query = shared.get("query", "")
            feedback = check_feedback_in_message(query)
            print(f"🔍 Processing feedback for report: '{feedback}'")
            
            if feedback == "approved":
                print("✅ Report approved - finalizing")
                session["waiting_for_feedback"] = None
                save_conversation(conversation_id, session)
                return "FEEDBACK_PROCESSED", "approved", shared["queue"]
            elif feedback == "rejected":
                print("❌ Report rejected - regenerating")
                session["waiting_for_feedback"] = None
                save_conversation(conversation_id, session)
                return "FEEDBACK_PROCESSED", "rejected", shared["queue"]
            else:
                print("❓ Unclear feedback - asking for clarification")
                session["waiting_for_feedback"] = None
                save_conversation(conversation_id, session)
                return "FEEDBACK_PROCESSED", "default", shared["queue"]
        
        # First time showing review - create review message
        self.log_to_flow(shared, None)  # Stop flow thoughts
        
        # Get the result to review
        context = self.get_research_context(shared)
        review_content = self.get_review_content(context)
        
        review_message = self.format_review_message(review_content)
        return review_message, shared["queue"]
    
    def exec(self, prep_res: Union[Tuple[str, str, Queue[str]], Tuple[str, Queue[str]]]) -> str:
        # Check if we're processing feedback
        if len(prep_res) == 3 and prep_res[0] == "FEEDBACK_PROCESSED":
            _, feedback_result, _ = prep_res
            return feedback_result
        
        # Normal review display - unpack 2-tuple
        if len(prep_res) == 2:
            review_message, queue = prep_res
            queue.put(review_message)
            queue.put(None)  # Signal UI completion - CRITICAL FOR UI TO NOT HANG!
            return review_message
        
        # Fallback - shouldn't reach here
        return "Error in review processing"
    
    def post(self, shared: SharedState, prep_res: Union[Tuple[str, str, Queue[str]], Tuple[str, Queue[str]]], exec_res: str) -> str:
        # Check if we processed feedback
        if len(prep_res) == 3 and prep_res[0] == "FEEDBACK_PROCESSED":
            return exec_res  # Return the routing decision
        
        # Normal first-time review - pause execution
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        session["waiting_for_feedback"] = self.get_review_type()
        save_conversation(conversation_id, session)
        
        return "pause"  # Route to PauseForFeedback node instead of "done"
    
    def get_review_content(self, context: ResearchContext) -> str:
        """Get the proposal to review from research context"""
        return context.get("proposal") or "No proposal generated"
    
    def format_review_message(self, content: str) -> str:
        """Format the review message for the final proposal"""
        return f"""📋 **Final Research Proposal:**

{content}

**Please review the proposal:**
👍 Say **"approved"** or **"ok"** to finalize and complete
👎 Say **"rejected"** or **"revise"** to improve the proposal"""
    
    def get_review_type(self) -> str:
        """Return the review type for feedback tracking"""
        return "report"
    
    def log_to_flow(self, shared: SharedState, message) -> None:
        """Helper to log messages to flow queue"""
        flow_log = shared["flow_queue"]
        flow_log.put(message)
    
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
            all_steps=journey,
            original_query=self._get_original_query(shared, journey)
        )
    
    def _find_step_result(self, journey, step_name):
        """Find the result of a specific step in the research journey"""
        for entry in reversed(journey):
            if entry.get("step") == step_name:
                return entry.get("result")
        return None
    
    def _get_original_query(self, shared, journey):
        """Get the original user query that started this research"""
        topic = self._find_step_result(journey, "topic_extraction")
        if topic:
            return topic
        return shared.get("query", "")
