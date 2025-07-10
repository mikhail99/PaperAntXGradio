from core.pocketflow_demo.nodes.actions import Action
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from queue import Queue
from pocketflow import Node

class ReviewQueries(Node):
    """Review node for search queries - presents results for human approval"""
    
    @staticmethod
    def required_params():
        return []  # Review nodes don't need input parameters
    
    @staticmethod
    def action_type():
        return Action.review_queries  # Proper Action enum value
    
    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)  # End flow thoughts

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        queries_result = session.get("action_result", "No queries generated")
        
        review_message = f"""🔍 **Generated Search Queries:**

{queries_result}

**Please review these queries:**
👍 **Like** to proceed with literature review
👎 **Dislike** to generate different queries"""
        
        return review_message, shared["queue"]

    def exec(self, prep_res):
        review_message, queue = prep_res
        queue.put(review_message)
        queue.put(None)  # End chat messages
        return review_message

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        session["waiting_for_feedback"] = "queries"
        save_conversation(conversation_id, session)
        return "done"

class ReviewReport(Node):
    """Review node for final report - presents results for human approval"""
    
    @staticmethod
    def required_params():
        return []  # Review nodes don't need input parameters
    
    @staticmethod
    def action_type():
        return Action.review_report  # Proper Action enum value
    
    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)  # End flow thoughts

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        report_result = session.get("action_result", "No report generated")
        
        review_message = f"""📋 **Final Research Proposal:**

{report_result}

**Please review the proposal:**
👍 **Like** to finalize and complete
👎 **Dislike** to revise the proposal"""
        
        return review_message, shared["queue"]

    def exec(self, prep_res):
        review_message, queue = prep_res
        queue.put(review_message)
        queue.put(None)  # End chat messages
        return review_message

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        session["waiting_for_feedback"] = "report"
        save_conversation(conversation_id, session)
        return "done"
