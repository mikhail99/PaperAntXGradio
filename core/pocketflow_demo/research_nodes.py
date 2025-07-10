from datetime import datetime
from queue import Queue

import yaml
from pocketflow import Node

from core.pocketflow_demo.utils.call_llm import call_llm
from core.pocketflow_demo.utils.conversation import load_conversation, save_conversation
from core.pocketflow_demo.utils.format_chat_history import format_chat_history
from core.pocketflow_demo.research_state import Action

def compute_next_action(history: list[dict], query: str, last_action: str, last_action_result: str) -> str:
    if last_action is None:
        next_action = Action.do_generate_queries
    elif last_action == Action.do_generate_queries:
        next_action = Action.do_literature_review
    elif last_action == Action.do_literature_review:
        next_action = Action.do_literature_review_gap
    elif last_action == Action.do_literature_review_gap:
        next_action = Action.do_write_proposal
    elif last_action == Action.do_write_proposal:
        next_action = Action.do_result_notification
    else:
        next_action = Action.do_follow_up
    decision = {
        "thinking": "I think so",
        "action": next_action,
        "reason": "I think so",
        "question": "...",
        "topic": "llm for math",
        "search_query": "llm for math research",
        "summary": "literature summary here",
        "gaps": "research gaps identified",
        "result": "final result here",
    }
    return decision

def prepare_for_next_action(session, decision: dict, flow_log):
    """Generic parameter preparation using node self-declaration"""
    next_action = decision["action"]
    
    # Initialize session params if it doesn't exist
    if "params" not in session:
        session["params"] = {}
    
    # Find the node class for this action
    if next_action not in ACTION_TO_NODE:
        # Fallback to follow-up for unknown actions
        return handle_missing_action(session, flow_log, next_action)
    
    node_class = ACTION_TO_NODE[next_action]
    required_params = node_class.required_params()
    
    # Extract and validate parameters
    params = {}
    missing_params = []
    
    for param in required_params:
        if param in decision:
            params[param] = decision[param]
        else:
            missing_params.append(param)
    
    if missing_params:
        # Fallback to follow-up with helpful message
        return handle_missing_params(session, flow_log, next_action, missing_params)
    
    # Store parameters and log success
    session["params"][next_action] = params
    flow_log.put(f"➡️ Agent decided to {next_action} with params: {params}")
    return next_action

def handle_missing_action(session, flow_log, action):
    """Handle unknown action by falling back to follow-up"""
    # Initialize session params if it doesn't exist
    if "params" not in session:
        session["params"] = {}
    
    question = f"I'm not sure how to handle the action '{action}'. Could you please rephrase your request?"
    session["params"][Action.do_follow_up] = {"question": question}
    flow_log.put(f"⚠️ Unknown action: {action}. Falling back to follow-up.")
    return Action.do_follow_up

def handle_missing_params(session, flow_log, action, missing_params):
    """Handle missing parameters by falling back to follow-up"""
    # Initialize session params if it doesn't exist
    if "params" not in session:
        session["params"] = {}
    
    question = f"I need more information to {action}. Missing: {', '.join(missing_params)}. Could you provide these details?"
    session["params"][Action.do_follow_up] = {"question": question}
    flow_log.put(f"⚠️ Missing parameters for {action}: {missing_params}. Falling back to follow-up.")
    return Action.do_follow_up

class ResearchAgentRouter(Node):
    @staticmethod
    def required_params():
        return []  # Router doesn't need input parameters
    
    @staticmethod
    def action_type():
        return None  # Router is not an action, it decides actions

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        return session, shared["history"], shared["query"]

    def exec(self, prep_res):
        session, history, query = prep_res
        last_action = session.get("last_action", None) 
        last_action_result = session.get("action_result", None)
        decision = compute_next_action(history, query, last_action, last_action_result)
        return decision

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["last_action"] = exec_res["action"]
        flow_log: Queue = shared["flow_queue"]

        for line in exec_res["thinking"].split("\n"):
            line = line.replace("-", "").strip()
            if line:
                flow_log.put(f"🤔 {line}")

        next_action = prepare_for_next_action(session, exec_res, flow_log)
        save_conversation(conversation_id, session)
        return next_action

class GenerateQueries(Node):
    @staticmethod
    def required_params():
        return ["topic"]
    
    @staticmethod
    def action_type():
        return Action.do_generate_queries

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        topic = session["params"][Action.do_generate_queries]["topic"]
        return topic

    def exec(self, prep_res):
        topic = prep_res
        return f"search queries for {topic}"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Generated queries: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"

class LiteratureReview(Node):
    @staticmethod
    def required_params():
        return ["search_query"]
    
    @staticmethod
    def action_type():
        return Action.do_literature_review

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        search_query = session["params"][Action.do_literature_review]["search_query"]
        return search_query

    def exec(self, prep_res):
        search_query = prep_res
        return f"Literature review result for {search_query}"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Literature review: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"

class SynthesizeGap(Node):
    @staticmethod
    def required_params():
        return ["summary"]
    
    @staticmethod
    def action_type():
        return Action.do_literature_review_gap

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        summary = session["params"][Action.do_literature_review_gap]["summary"]
        return summary

    def exec(self, prep_res):
        summary = prep_res
        return f"Research gaps identified for {summary}"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Gap analysis: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"

class ReportGeneration(Node):
    @staticmethod
    def required_params():
        return ["gaps"]
    
    @staticmethod
    def action_type():
        return Action.do_write_proposal

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        gaps = session["params"][Action.do_write_proposal]["gaps"]
        return gaps

    def exec(self, prep_res):
        gaps = prep_res
        return f"Project proposal for {gaps}"

    def post(self, shared, prep_res, exec_res):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(f"⬅️ Proposal generated: {exec_res}")

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        return "default"
    
class FollowUp(Node):
    @staticmethod
    def required_params():
        return ["question"]
    
    @staticmethod
    def action_type():
        return Action.do_follow_up

    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        question = session["params"][Action.do_follow_up]["question"]
        return question, shared["queue"]

    def exec(self, prep_res):
        question, queue = prep_res
        queue.put(question)
        queue.put(None)
        return question

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = exec_res
        return "done"

class ResultNotification(Node):
    @staticmethod
    def required_params():
        return ["result"]
    
    @staticmethod
    def action_type():
        return Action.do_result_notification

    def prep(self, shared):
        flow_log: Queue = shared["flow_queue"]
        flow_log.put(None)

        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        result = session["params"][Action.do_result_notification]["result"]
        return result, shared["queue"]

    def exec(self, prep_res):
        result, queue = prep_res
        queue.put(result)
        queue.put(None)
        return result

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session: dict = load_conversation(conversation_id)
        session["action_result"] = None
        session["last_action"] = None
        save_conversation(conversation_id, session)
        return "done"

# Action to Node Registry
ACTION_TO_NODE = {
    Action.do_generate_queries: GenerateQueries,
    Action.do_literature_review: LiteratureReview,
    Action.do_literature_review_gap: SynthesizeGap,
    Action.do_write_proposal: ReportGeneration,
    Action.do_follow_up: FollowUp,
    Action.do_result_notification: ResultNotification,
}
